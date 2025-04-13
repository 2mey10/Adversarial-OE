import copy
import json
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from pytorch_ood.loss import OutlierExposureLoss
from pathlib import Path
from logic.attacks import generate_adversarial_outlier
from logic.utils import (
    get_model, 
    get_device,
    prepare_data_loaders,
    create_optimizer_and_scheduler,
    create_dataframe_from_args,
    add_experiment_results,
    compute_accuracy,
    compute_test_auroc,
    compute_adversarial_auroc,
    create_auroc_metrics
)


def run(args):
    torch.manual_seed(1)
    np.random.seed(1)

    # read in arguments
    print(OmegaConf.to_yaml(args))
    # print keys and vars
    print(args.keys())
    print(vars(args))

    # ----------- model training program -----------

    # GPU acceleration
    device = get_device()
    print("Using device:", device)

    model = get_model(args)

    # define data loaders
    train_loader_inlier, train_loader_outlier = prepare_data_loaders(args)

    # define optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, train_loader_inlier)

    loss_fn = OutlierExposureLoss()

    # produce metrics before start
    start_auroc = create_auroc_metrics(model=model, args=args, adversarial=False)
    print("AUROC before training: ", start_auroc)

    def train_adversarial_default(epoch):
        loss_avg = 0.0
        model.train()  # enter train mode
        
        # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
        # When using GAN Images as outliers, the offset is not used, as this somehow leads to a crash
        if args.dataset_out != "GAN_IMG":
            train_loader_outlier.dataset.offset = np.random.randint(len(train_loader_outlier.dataset))
        print("Starting adversarial training epoch %d" % epoch)
        print("Length of train_loader_inlier: ", len(train_loader_inlier))
        print("Length of train_loader_outlier: ", len(train_loader_outlier))

        for in_set, out_set in tqdm(zip(train_loader_inlier, train_loader_outlier), total=len(train_loader_inlier)):
            # model.train()  # enter train mode
            # define variables
            inlier = in_set[0]
            outlier = out_set[0]
            target_inlier = in_set[1]
            target_outlier = out_set[1]

            # stop if we have covered the entire outlier dataset and we have remains (10000%64=16 -> the code would crash)
            if len(outlier) < args.oe_batch_size:
                break

            # create a copy of the model
            model_copy = copy.deepcopy(model)
            
            # # perform adversarial attack on outliers
            perturbed_outlier = generate_adversarial_outlier(args, model_copy, inlier, outlier, target_inlier,
                                                             target_outlier, args.eps_oe, device)

            perturbed_outlier.detach()
            model.zero_grad()

            # concatenate inlier and perturbed outlier
            data = torch.cat((inlier.to(device), perturbed_outlier.to(device)), 0).to(device)

            # forward
            data = data.float()
            x = model(data)

            # backward
            optimizer.zero_grad()

            # calculate loss
            loss = loss_fn(x, torch.cat((target_inlier.to(device), target_outlier.to(device)), 0))

            # update parameters
            loss.backward()
            optimizer.step()

            scheduler.step()
            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        train_loss = loss_avg

        if hasattr(args, 'train_auroc') and args.train_auroc:
            print("Calculating training AUROC")
            auroc = create_auroc_metrics(model=model, args=args, adversarial=False)
        else:
            auroc = "empty"

        return train_loss,auroc
    
    def run_training_loop(args):
        """
        Runs the training loop for the given model and dataset.

        Args:
            args (object): The arguments for training.
            model (object): The model to be trained.
            device (str): The device to be used for training.

        Returns:
            tuple: A tuple containing the losses and aurocs_train.

        Raises:
            ValueError: If the dataset is unknown.
        """
        training_start = time.time()
        losses = []
        aurocs_train = []

        if args.dataset_in in ["cifar10", "cifar100"]:
            for epoch in range(args.epochs):
                begin_epoch = time.time()

                train_loss_epoch, auroc_train = train_adversarial_default(epoch)
                losses.append({"epoch": epoch, "loss": round(train_loss_epoch, 4)})
                aurocs_train.append({"epoch": epoch, "auroc": auroc_train})

                print(
                    f'Epoch {epoch + 1:3d} | Time {int(time.time() - begin_epoch):5d} | Train Loss {train_loss_epoch:.4f}')

        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset_in))

        time_needed = time.time() - training_start
        print(f"Finished training in {time_needed}")

        return losses, aurocs_train
    
    def run_training(model, device, args):

        # ----------- run training loop -----------
        accuracy_dict = compute_accuracy(model, device, args)
        print("Accuracy before OE: ", accuracy_dict)
        
        losses, aurocs_train = run_training_loop(args)

        # save aurcos to json file
        now = datetime.now()

        # Format as string: YYYYMMDD_HHMMSS
        id = now.strftime("%Y%m%d_%H%M%S")

        # Save aurocs to json file
        Path("training_aurocs").mkdir(parents=True, exist_ok=True)
        with open(f"training_aurocs/{id}.json", "w") as f:
            json.dump(aurocs_train, f)

        # ----------- save model -----------
        #save_model(model, architecture_folder)

        # ----------- compute Accuracy -----------
        accuracy_dict = compute_accuracy(model, device, args)

        # ----------- compute testing AUROC -----------
        if args.test_auroc:
            aurocs_test = create_auroc_metrics(model=model, args=args, adversarial=False)
        else:
            aurocs_test = {"empty": "empty"}

        # Save aurocs to json file
        Path("test_aurocs").mkdir(parents=True, exist_ok=True)
        with open(f"test_aurocs/{id}.json", "w") as f:
            json.dump(aurocs_test, f)

        # ----------- compute adversarial AUROC -----------
        # create a random id between 0 and 1000000 to save the json file

        if args.adversarial_test_auroc:
            aurocs_adversarial = create_auroc_metrics(model=model, args=args, adversarial=True)
        else:
            aurocs_adversarial = {"empty": "empty"}

        # Save aurocs to json file
        Path("adversarial_aurocs").mkdir(parents=True, exist_ok=True)
        with open(f"adversarial_aurocs/{id}.json", "w") as f:
            json.dump(aurocs_adversarial, f)

        # ----------- add results to dataframe -----------

        #create dataframe
        df = create_dataframe_from_args(args)

        # append the random_id to the dataframe
        df["save_id"] = id

        # save accuracy to dataframe
        add_experiment_results(df, results=accuracy_dict, column_name="accuracy")
        add_experiment_results(df, results=aurocs_train, column_name="auroc_train")
        add_experiment_results(df, results=aurocs_test, column_name="auroc_test")
        add_experiment_results(df, results=aurocs_adversarial, column_name="auroc_adversarial")
        add_experiment_results(df, results=losses, column_name="loss")
        df["auroc_adversarial_path"] = f"adversarial_aurocs/{id}.json"
        df["train_auroc_path"] = f"training_aurocs/{id}.json"
        df["test_auroc_path"] = f"test_aurocs/{id}.json"
        

        # read in the current dataframe that already contains the results from the previous experiments
        try:
            df_old = pd.read_excel("results.xlsx")
            print("found results.xlsx")
        except:
            df_old = pd.DataFrame() # create empty dataframe if no results.xlsx exists
        
        # append the new results to the old dataframe
        df = df_old._append(df, ignore_index=True)
        print("appended new results to dataframe")

        # save the dataframe to a new results.xlsx file
        df.to_excel("results.xlsx", index=False)
        print("saved dataframe to results.xlsx")

    # now run the program :)
    run_training(model, device, args)

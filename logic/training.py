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
            
            # perform adversarial attack on outliers
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

    def train_adversarial_imagenet(iters):
        loss_avg = 0.0
        # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
        # When using GAN Images as outliers, the offset is not used, as this somehow leads to a crash
        if args.dataset_out != "GAN_IMG":
            train_loader_outlier.dataset.offset = np.random.randint(len(train_loader_outlier.dataset))
        print("Starting adversarial imagenet training for %d iterations" % iters)
        print("Length of train_loader_inlier: ", len(train_loader_inlier))
        print("Length of train_loader_outlier: ", len(train_loader_outlier))

        it_train_in = iter(train_loader_inlier)
        it_train_out = iter(train_loader_outlier)
        bar = tqdm(range(iters))

        mav = 0.0
        mavs = []
        aurocs = []

        for i in bar:
            model.train()  # enter train mode
            try:
                inlier,target_inlier = next(it_train_in)
                outlier,target_outlier = next(it_train_out)
            except:
                it_train_in = iter(train_loader_inlier)
                it_train_out = iter(train_loader_outlier)
                
            # stop if we have covered the entire outlier dataset and we have remains (10000%64=16 -> the code would crash)
            if len(outlier) < args.oe_batch_size:
                continue

            # perform adversarial attack on outliers
            perturbed_outlier = generate_adversarial_outlier(args, model, inlier, outlier, target_inlier,
                                                             target_outlier, args.eps_oe, device)

            # concatenate inlier and perturbed outlier
            data = torch.cat((inlier.to(device), perturbed_outlier.to(device)), 0).to(device)

            # forward
            data = data.float()
            y_hat = model(data)

            # backward
            optimizer.zero_grad()

            # calculate loss
            loss = loss_fn(y_hat, torch.cat((target_inlier.to(device), target_outlier.to(device)), 0))

            # update parameters
            loss.backward()
            optimizer.step()

            scheduler.step()
            # exponential moving average
            mav = 0.2 * loss.item() + 0.8 * mav
            mavs.append(mav)

            bar.set_postfix({"loss": mav})
            if i % 100 == 0:
                # Check if args.train_auroc exists, else set it to False by default
                if hasattr(args, 'train_auroc') and args.train_auroc:
                    auroc = create_auroc_metrics(model=model, args=args, adversarial=False)
                else:
                    auroc = "empty"
                aurocs.append({
                    "loss": mav,
                    "auroc": auroc
                })
                print(f"AUROC for iteration {i}: {auroc}")

        return mavs,aurocs
    
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

        elif args.dataset_in == "imagenet":
            losses, aurocs_train = train_adversarial_imagenet(iters=args.iter)

        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset_in))

        time_needed = time.time() - training_start
        print(f"Finished training in {time_needed}")

        return losses, aurocs_train
    
    def run_training(model, device, args):

        # ----------- run training loop -----------
        # one accuracy for the fans :D
        accuracy_dict = compute_accuracy(model, device, args)
        print("Accuracy before OE: ", accuracy_dict)
        
        losses, aurocs_train = run_training_loop(args)

        # save aurcos to json file
        now = datetime.now()

        # Format as string: YYYYMMDD_HHMMSS
        train_id = now.strftime("%Y%m%d_%H%M%S")

        # Save aurocs to json file
        Path("training_aurocs").mkdir(parents=True, exist_ok=True)
        with open(f"training_aurocs/{train_id}.json", "w") as f:
            json.dump(aurocs_train, f)

        # ----------- save model -----------
        #save_model(model, architecture_folder)

        # ----------- compute Accuracy -----------
        accuracy_dict = compute_accuracy(model, device, args)
        #save_accuracy_to_file(accuracy_dict, architecture_folder)

        # ----------- compute testing AUROC -----------
        if args.test_auroc:
            aurocs_test = create_auroc_metrics(model=model, args=args, adversarial=False)
        else:
            aurocs_test = {"empty": "empty"}

        # ----------- compute adversarial AUROC -----------
        # create a random id between 0 and 1000000 to save the json file
        # Get current date and time
        now = datetime.now()

        # Format as string: YYYYMMDD_HHMMSS
        id = now.strftime("%Y%m%d_%H%M%S")

        if args.adversarial_test_auroc:
            aurocs_adversarial = create_auroc_metrics(model=model, args=args, adversarial=True)
        else:
            aurocs_adversarial = {"empty": "empty"}

        # Save aurocs to json file
        Path("adversarial_aurocs").mkdir(parents=True, exist_ok=True)
        with open(f"adversarial_aurocs/{id}.json", "w") as f:
            json.dump(aurocs_adversarial, f)
        
        # then append the random_id to the dataframe, so we can actually backtrack the results to the corresponding training
        # this gets done later
        
        # ----------- save results -----------
        #save_auroc_to_file(aurocs_train, aurocs_test, aurocs_adversarial, architecture_folder)

        # ----------- add results to dataframe -----------

        #create dataframe
        df = create_dataframe_from_args(args)

        # append the random_id to the dataframe
        df["save_id"] = id
        df["train_id"] = train_id

        # save accuracy to dataframe
        add_experiment_results(df, results=accuracy_dict, column_name="accuracy")
        add_experiment_results(df, results=aurocs_train, column_name="auroc_train")
        add_experiment_results(df, results=aurocs_test, column_name="auroc_test")
        add_experiment_results(df, results=aurocs_adversarial, column_name="auroc_adversarial")
        add_experiment_results(df, results=losses, column_name="loss")

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

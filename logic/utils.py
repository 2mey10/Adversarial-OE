import datetime
import time
import numpy as np
import pandas as pd
import pytorch_ood
import torch
import torchvision.models
import torchvision.transforms as trn
import torchvision.datasets as dset
from pytorch_ood.dataset.img import TinyImages300k
import os
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import ToRGB
from torch.utils.data import TensorDataset
from torchvision.models import resnet101
from torchvision.utils import save_image
from torch.utils.data import random_split, Subset
import torch.nn.functional as F
import json
from omegaconf import OmegaConf

from logic.eval import  select_first_n_classes_cifar100,  create_auroc_metrics
# fix the pytorch dataset certificate error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_transforms(dataset_name, img_size=32):
    resize = trn.Resize((img_size, img_size))

    if dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset_name == "cifar100":
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = trn.Compose([
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(img_size, padding=4),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])

    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean, std),
        resize
    ])

    return transform, test_transform


def get_training_datasets(args):
    """
    This function returns the training datasets for the in-distribution and the out-of-distribution data.

    :param args: the arguments passed to the main function
    :return: the training datasets for the in-distribution and the out-of-distribution data
    """
    # check if inlier dataset is supported
    if args.dataset_in not in ["cifar10", "cifar100"]:
        raise ValueError(f"Unsupported dataset: {args.dataset_in}")

    # check if outlier dataset is supported
    if args.dataset_out not in ["300K", "GAN_IMG"]:
        raise ValueError(f"Unsupported dataset: {args.dataset_out}")

    # check for dataset_in to prepare inliers
    if args.dataset_in == 'cifar10':
        train_transform, _ = get_transforms("cifar10",32)
        train_data_in = dset.CIFAR10('datasets', train=True, transform=train_transform, download=True)

    if args.dataset_in == "cifar100":
        train_transform, _ = get_transforms("cifar100",32)
        train_data_in = dset.CIFAR100('datasets', train=True, transform=train_transform, download=True)

        if hasattr(args, 'n_classes') and args.n_classes < 100:
            train_data_in = select_first_n_classes_cifar100(train_data_in, args.n_classes)

    # check for dataset_out to prepare outliers
    if args.dataset_out == '300K':
        train_transform, _ = get_transforms("cifar10",32)
        train_data_out = TinyImages300k(root="datasets", transform=train_transform,
                                        target_transform=pytorch_ood.utils.ToUnknown(), download=True)

    if args.dataset_out == "GAN_IMG":
        # print(f"LOG: Loading GAN images for {args.dataset_in}... ")
        if args.dataset_in in ["cifar10", "cifar100"]:
            # use args.gan_sigma for the gan_file_name
            gan_file_name = f"samples-{args.gan_sigma}.npz"
            gan_path = os.path.join(os.getcwd(), 'datasets', gan_file_name)
            gan_data_numpy = np.load(gan_path)

            gan_data = torch.from_numpy(gan_data_numpy['x'])

            # normalize gan images and set the target to -1
            train_data_out = [(el / 255, -1) for el in gan_data]
            train_data_out = random_split(train_data_out,[args.n_out_images, len(train_data_out)-args.n_out_images])[0]
            # artificially makes this 50000 long, copy the train_data_out so often as needed til the len is 50k
            while len(train_data_out) < 50000:
                train_data_out = train_data_out + train_data_out
            train_data_out = Subset(train_data_out, range(50000))

    return train_data_in, train_data_out

    
def get_test_dataset(args):
    if args.dataset_in == "cifar10":
        _, test_transform = get_transforms(args.dataset_in)
        return dset.CIFAR10(root="datasets", train=False, transform=test_transform, download=True)

    elif args.dataset_in == "cifar100":
        _, test_transform = get_transforms(args.dataset_in)
        return dset.CIFAR100(root="datasets", train=False, transform=test_transform, download=True)

    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset_in))

def get_model(args):
    """
    This function returns the model for the in-distribution data.

    :param args: the arguments passed to the main function
    :return: the model for the in-distribution data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == "wrn":
        if args.dataset_in == "cifar10":
            model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device).eval()
        if args.dataset_in == "cifar100":
            model = WideResNet(num_classes=100, pretrained="cifar100-pt").to(device).eval()
    else:
        raise ValueError("This model is not defined and implemented!")

    return model


def save_altered_images(outlier, perturbed_outlier, epoch, architecture_folder):
    """
    This function saves the original and the perturbed outlier images.

    :param outlier: the original outlier image
    :param perturbed_outlier: the perturbed outlier image
    :param epoch: the current epoch
    :param architecture_folder: the folder where the images should be saved
    """
    image_save_folder = os.path.join(architecture_folder, str(epoch))
    try:
        os.mkdir(image_save_folder)
    except:
        None
    save_image(outlier[0], f"{image_save_folder}/orig.png")
    save_image(perturbed_outlier[0], f"{image_save_folder}/perturbed.png")


def get_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for computations.")
        return torch.device('cuda')
    else:
        print("CUDA is not available. Using CPU for computations.")
        return torch.device('cpu')
    

def create_data_loader(data, batch_size, shuffle, num_workers=0, pin_memory=True):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, pin_memory=pin_memory)

def prepare_data_loaders(args):
    # Retrieve logic data
    train_data_in, train_data_out = get_training_datasets(args)

    # Define data loaders
    train_loader_in = create_data_loader(train_data_in, args.batch_size, shuffle=True)
    train_loader_out = create_data_loader(train_data_out, args.oe_batch_size, shuffle=False)

    return train_loader_in, train_loader_out


def create_optimizer_and_scheduler(model, args, train_loader_in):
    # define logic parameters
    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    optimizer = torch.optim.SGD(
        model.parameters(), args.learning_rate, momentum=args.momentum,
        weight_decay=args.decay, nesterov=True)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    return optimizer, scheduler


def compute_accuracy(model, device, args):
    model = model.to(device).eval()
    test_set_dataloader = torch.utils.data.DataLoader(
        get_test_dataset(args),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    correct = 0
    with torch.no_grad():
        for data, target in test_set_dataloader:
            data, target = data.to(device), target.to(device)

            # forward
            output = F.softmax(model(data), dim=1)

            # accuracy
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()

    accuracy = correct / len(test_set_dataloader.dataset)
    print(f"accuracy: {accuracy}")
    print(f"percentage of false predictions: {1 - accuracy}")
    return accuracy

def save_accuracy_to_file(accuracy_dict, architecture_folder):

    # write results to file
    accuracy_path = os.path.join(architecture_folder, "accuracy")
    with open(f"{accuracy_path}.txt", 'w') as json_file:
        json.dump(accuracy_dict, json_file)

    print(f"Dumped accuracy data to {accuracy_path}.txt")

def compute_adversarial_auroc(args, model):
    if hasattr(args, "adversarial_test_auroc") and args.adversarial_test_auroc:
        aurocs_adversarial = create_auroc_metrics(model=model,args=args, adversarial=True)
    else:
        aurocs_adversarial = {
            "aurocs_adversarial": "empty"
        }

    print(f"Adversarial AUROC: {aurocs_adversarial}")
    return aurocs_adversarial


def compute_test_auroc(args, model):
    if hasattr(args, 'test_auroc') and args.test_auroc:
        aurocs_test = create_auroc_metrics(model=model, args=args, adversarial=False)
    else:
        aurocs_test = {
            "aurocs_test": "empty"
        }

    print(f"Test AUROC: {aurocs_test}")
    return aurocs_test

def save_auroc_to_file(aurocs_train, aurocs_test, aurocs_adversarial, architecture_folder):
    auroc_dict = {
        "aurocs_train": aurocs_train,
        "aurocs_test": aurocs_test,
        "aurocs_adversarial": aurocs_adversarial
    }

    print(auroc_dict)

    results_path = os.path.join(architecture_folder, "results")
    with open(f"{results_path}.txt", 'w') as json_file:
        json.dump(auroc_dict, json_file)

    print(f"Dumped data to {results_path}.txt")


def save_model(model, architecture_folder):
    model_path = os.path.join(architecture_folder, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

def create_dataframe_from_args(args):
    # Convert DictConfig to dictionary
    args_dict = OmegaConf.to_container(args, resolve=True)

    # Create DataFrame
    df = pd.DataFrame([args_dict])

    return df

def add_experiment_results(df, results,column_name):
    # dump results to json string
    results_json = json.dumps(results)

    # add results to DataFrame
    df[column_name] = results_json

    return df
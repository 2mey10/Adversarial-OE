from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as trn
import torchvision.datasets as dset

from pytorch_ood.detector import MaxSoftmax, EnergyBased
from torch.utils.data import Dataset, DataLoader
import pytorch_ood
from pytorch_ood.utils import OODMetrics, ToRGB, ToUnknown
from pytorch_ood.dataset.img import Textures, TinyImageNetCrop, TinyImageNetResize, LSUNCrop, LSUNResize, GaussianNoise, \
    UniformNoise, PixMixDataset, FoolingImages, ImageNetR, ImageNetO, ImageNetA, iNaturalist, NINCO
from tqdm import tqdm
import torchattacks
from torch.utils.data import Subset
from copy import deepcopy

def get_cifar_test_transform():
    img_size = 32
    resize = trn.transforms.Resize((img_size, img_size))
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    return trn.Compose([pytorch_ood.utils.ToRGB(), trn.ToTensor(), trn.Normalize(mean, std), resize])

def get_imagenet_rgb_transform():
    imagenet_transform = torchvision.models.Wide_ResNet50_2_Weights.DEFAULT.transforms()
    return trn.Compose([ToRGB(),imagenet_transform])

def select_first_n_classes_cifar100(cifar100_dataset, n_classes):
    """
    Selects the first n_classes from CIFAR-100 dataset.

    Parameters:
    - cifar100_dataset: CIFAR-100 dataset.
    - n_classes: Number of classes to select.

    Returns:
    - selected_dataset: Subset of the original CIFAR-100 dataset with the first n_classes.
    """

    # Get the total number of classes in CIFAR-100
    total_classes = 100

    # Check if n_classes is valid
    if n_classes <= 0 or n_classes > total_classes:
        raise ValueError("Invalid value for n_classes. It should be between 1 and 100.")

    # Select the first n_classes
    selected_classes = range(n_classes)

    # Filter the dataset to keep only the selected classes
    selected_indices = [i for i, (_, label) in enumerate(cifar100_dataset) if label in selected_classes]
    selected_dataset = Subset(cifar100_dataset, selected_indices)

    return selected_dataset

def create_known_dataset(args):
    dataset_to_test = args.dataset_in
    test_transform = get_cifar_test_transform()
    if dataset_to_test == "cifar10":
        print("Creating CIFAR-10 dataset for evaluation")
        return dset.CIFAR10('datasets', train=False, transform=test_transform, download=True)
    if dataset_to_test == "cifar100":
        if hasattr(args, 'n_classes') and args.n_classes < 100:
            print(f"Creating CIFAR-100 dataset for evaluation with {args.n_classes} classes")
            return select_first_n_classes_cifar100(dset.CIFAR100('datasets', train=False, transform=test_transform,
                                                                 download=True), args.n_classes)
        print("Creating CIFAR-100 dataset for evaluation")
        return dset.CIFAR100('datasets', train=False, transform=test_transform, download=True)

    if dataset_to_test == "imagenet":
        print("Creating ImageNet dataset for evaluation")
        test_transform = torchvision.models.Wide_ResNet50_2_Weights.DEFAULT.transforms()
        return dset.ImageNet(root=args.imagenet_path, split='val', transform=test_transform)
        #return dset.ImageNet(root='/nfs1/kirchhei/imagenet-2012/', split='val', transform=test_transform)


def calculate_ood_metrics(dataset_loader, dataset_name, model,adversarial=False):
    softmax = MaxSoftmax(model)
    energy = EnergyBased(model)
    metrics = OODMetrics()
    metrics_energy = OODMetrics()

    if adversarial:
        print(f"Running adversarial attacks on {dataset_name}")
        softmax_metrics = []
        energy_metrics = []
        #for eps in [0,0.1]:
        for eps in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            for x, y in tqdm(dataset_loader):
                # manipualte x using torchattacks fgsm
                # compute fgsm without library
                x_adv = Variable(x.data, requires_grad=True)
                logits = model(x_adv.cuda())
                loss = torch.nn.CrossEntropyLoss()(logits, torch.ones(x.shape[0]).long().cuda())
                grad = torch.autograd.grad(loss, x_adv,retain_graph=False, create_graph=False)[0]
                x_adv = x_adv + eps * grad.sign()

                with torch.no_grad():
                    logits = model(x_adv.cuda())
                    metrics.update(softmax.score(logits), y)
                    metrics_energy.update(energy.score(logits), y)

            m_softmax = metrics.compute()
            m_softmax.update({
                "Dataset": dataset_name,
                "Method": "Adversarial Softmax",
                "Epsilon": eps
            })

            m_energy = metrics_energy.compute()
            m_energy.update({
                "Dataset": dataset_name,
                "Method": "Adversarial Energy",
                "Epsilon": eps
            })

            softmax_metrics.append(m_softmax)
            energy_metrics.append(m_energy)

            print(f"Softmax AUROC for eps {eps} -> {m_softmax['AUROC']:.3%}")
            print(f"Energy AUROC for eps {eps} -> {m_energy['AUROC']:.3%}")


        return softmax_metrics, energy_metrics

    else:
        print("Running normal evaluation, no adversarial attacks")
        with torch.no_grad():
            for x, y in tqdm(dataset_loader):
                logits = model(x.cuda())
                metrics.update(softmax.score(logits), y)
                metrics_energy.update(energy.score(logits), y)

        m_softmax = metrics.compute()
        m_softmax.update({
            "Dataset": dataset_name,
            "Method": "Softmax"
        })

        m_energy = metrics_energy.compute()
        m_energy.update({
            "Dataset": dataset_name,
            "Method": "Energy"
        })

        return m_softmax, m_energy


def create_auroc_metrics(args,model, adversarial=False):
    print(f"Creating metrics for dataset {args.dataset_in}")
    model.eval()
    model.cuda()

    dataset_in_test = create_known_dataset(args)

    ood_datasets = get_ood_sets(args.dataset_in)

    softmax_metrics = []
    energy_metrics = []

    print(f"Running evaluation for dataset {args.dataset_in}")
    for dataset_name, dataset in ood_datasets.items():
        print("Testing dataset", dataset_name)
        dataset_loader = DataLoader(dataset + dataset_in_test, batch_size=128, num_workers=1)
        m_softmax, m_energy = calculate_ood_metrics(dataset_loader, dataset_name, model,adversarial=adversarial)
        softmax_metrics.append(m_softmax)
        energy_metrics.append(m_energy)

        #fallback mode, as adversarial attacks completely ruin this code xd
        try:
            print(f"AUROC   -> {m_softmax['AUROC']:.3%}")
        except Exception as e:
            pass

    model.train()  # enter train mode

    try:
        # average metrics
        metrics_sum_softmax = {"AUROC": 0, "AUPR-IN": 0, "AUPR-OUT": 0, "FPR95TPR": 0}
        metrics_sum_energy = {"AUROC": 0, "AUPR-IN": 0, "AUPR-OUT": 0, "FPR95TPR": 0}

        for metric in softmax_metrics:
            for key in metrics_sum_softmax.keys():
                metrics_sum_softmax[key] += metric[key]

        for metric in energy_metrics:
            for key in metrics_sum_energy.keys():
                metrics_sum_energy[key] += metric[key]

        metrics_average_softmax = {key: value / len(softmax_metrics) for key, value in metrics_sum_softmax.items()}
        metrics_average_energy = {key: value / len(energy_metrics) for key, value in metrics_sum_energy.items()}

        # Calculate the averages
        softmax_metrics.append({
            "Dataset": "Average Softmax AUROC",
            "metrics": metrics_average_softmax
        })

        energy_metrics.append({
            "Dataset": "Average Energy AUROC",
            "metrics": metrics_average_energy
        })
        print("Average Softmax:",metrics_average_softmax)
        print("Average Energy:",metrics_average_energy)
    except Exception as e:
        print(f"Error in calculating average metrics: {e}")
        pass

    return [softmax_metrics, energy_metrics]


def get_ood_sets(dataset_in):
    print(f"Getting OOD datasets for {dataset_in}")
    if dataset_in == "imagenet":
        test_transform = get_imagenet_rgb_transform()
        return {
        "ImageNetO": ImageNetO(root="/nfs1/botschen/datasets/",
                               download=True, transform=test_transform, target_transform=ToUnknown()),
        "iNaturalist": iNaturalist(root="/nfs1/botschen/datasets/",
                                    download=True, transform=test_transform, target_transform=ToUnknown()),
        "NINCO": NINCO(root="/nfs1/botschen/datasets/",
                        download=True, transform=test_transform, target_transform=ToUnknown())
    }

    if dataset_in in ["cifar10","cifar100"]:
        test_transform = get_cifar_test_transform()
        return {
            "textures": Textures('datasets', transform=test_transform, target_transform=ToUnknown(), download=True),
            "lsun resize": LSUNResize('datasets', transform=test_transform, target_transform=ToUnknown(),
                                      download=True),
            "lsun crop": LSUNCrop('datasets', transform=test_transform, target_transform=ToUnknown(), download=True),
            "gaussian noise": GaussianNoise(length=500, transform=test_transform, target_transform=ToUnknown()),
            "uniform noise": UniformNoise(length=500, transform=test_transform, target_transform=ToUnknown()),
            "tinyimagenet resize": TinyImageNetResize('datasets', transform=test_transform,
                                                      target_transform=ToUnknown(), download=True),
            "tinyimagenet crop": TinyImageNetCrop('datasets', transform=test_transform, target_transform=ToUnknown(),
                                                  download=True),
            "fooling images": FoolingImages('datasets', transform=test_transform, target_transform=ToUnknown(),
                                            download=True)
        }
    
    raise ValueError(f"Unknown dataset {dataset_in}")
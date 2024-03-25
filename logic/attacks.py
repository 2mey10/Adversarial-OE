import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_ood.loss import OutlierExposureLoss


def prepare_images(inlier, outlier, target_inlier, target_outlier, device):
    img_inlier = inlier.clone().detach().to(device)
    img_outlier = outlier.clone().detach().to(device)
    img_inlier.requires_grad = True
    img_outlier.requires_grad = True
    targets_concat = torch.cat((target_inlier.to(device), target_outlier.to(device)), 0)

    return img_inlier, img_outlier, targets_concat

def fgsm_softmax(model, inlier, outlier, target_inlier, target_outlier, eps_oe, device="gpu", log=False):
    """
    FGSM attack for Outlier Exposure

    Args:
        model: model to attack
        inlier: input data from inlier dataset
        outlier: input data from outlier dataset
        target_inlier: target data from inlier dataset
        target_outlier: target data from outlier dataset
        eps_oe: epsilon for the attack
        device: device to use for the attack


    """

    # Prepare images
    inlier, outlier, targets_concat = prepare_images(inlier, outlier, target_inlier, target_outlier, device)
    images_concat = torch.cat((inlier, outlier))

    # Get outputs of the model on inlier and outlier images
    outputs = model(images_concat)

    # Calculate loss
    criterion = OutlierExposureLoss()
    loss = criterion(outputs, targets_concat)

    # Calculate gradients of model in the direction of img_inlier (somehow this works better than using the outlier gradient?)
    grad = torch.autograd.grad(loss, images_concat,
                               retain_graph=False, create_graph=False)[0]

    # Update adversarial images
    try:
        adv_images = outlier + (eps_oe * grad.sign()[len(outlier):])

        if log:
            print(f"eps: {eps_oe} | Norm:{torch.linalg.norm(adv_images.cpu() - outlier.cpu()) / len(adv_images)}")
    except Exception as e:
        print("Error in FGSM attack: ", e)
        adv_images = outlier.clone().detach()


    return adv_images

def epsilon_gaussian_noise(model, inlier, outlier, target_inlier, target_outlier, eps_oe, device="gpu", log=False):
    """
    Gradient Method attack for Outlier Exposure

    Args:
        model: model to attack
        inlier: input data from inlier dataset
        outlier: input data from outlier dataset
        target_inlier: target data from inlier dataset
        target_outlier: target data from outlier dataset
        eps_oe: epsilon for the attack
        device: device to use for the attack


    """

    # Add Gaussian noise to the image
    var = 1
    sigma = var ** 0.5
    noisy_list = []
    gaussian = np.random.normal(0, sigma, (
        outlier[0].shape[0], outlier[0].shape[1], outlier[0].shape[2]))
    gaussian = gaussian.reshape(outlier[0].shape[0], outlier[0].shape[1], outlier[0].shape[2])

    for image in outlier:
        noisy_image = image.cpu() + eps_oe * gaussian
        noisy_list.append(noisy_image)

    adv_images = torch.stack(noisy_list)

    model.zero_grad()

    return adv_images

def energy_attack(model, inlier, outlier, target_inlier, target_outlier, eps_oe, device="gpu", log=False):
    """
    FGSM attack with Energy loss for Outlier Exposure

    Args:
        model: model to attack
        inlier: input data from inlier dataset
        outlier: input data from outlier dataset
        target_inlier: target data from inlier dataset
        target_outlier: target data from outlier dataset
        eps_oe: epsilon for the attack
        device: device to use for the attack


    """

    # Prepare images
    inlier, outlier, targets_concat = prepare_images(inlier, outlier, target_inlier, target_outlier, device)


    # Get outputs of the model on inlier and outlier images
    outputs = model(outlier)

    # Calculate energy on the outlier batch
    energy = -torch.logsumexp(outputs, dim=1)

    # Calculate mean energy
    energy_mean = energy.mean()

    # Calculate gradients of model energy w.r.t. outlier
    grad_energy = torch.autograd.grad(energy_mean, outlier,
                                      retain_graph=False, create_graph=False)[0]

    # Update adversarial images
    try:
        adv_images = outlier - (eps_oe * grad_energy.sign())

        if log:
            print(f"eps: {eps_oe} | Norm:{torch.linalg.norm(adv_images.cpu() - outlier.cpu()) / len(adv_images)}")
    except Exception as e:
        print("Error in FGSM attack: ", e)
        adv_images = outlier.clone().detach()

    return adv_images

def pgd_softmax(model, inlier, outlier, target_inlier, target_outlier, eps_oe, steps_oe, alpha_oe, device="gpu", log=False):
    """
    PGD attack for Outlier Exposure

    Args:
        model: model to attack
        inlier: input data from inlier dataset
        outlier: input data from outlier dataset
        target_inlier: target data from inlier dataset
        target_outlier: target data from outlier dataset
        eps_oe: epsilon for the attack
        steps_oe: number of steps for the attack
        alpha_oe: step size for the attack
        device: device to use for the attack
    """

    if eps_oe == 0:
        return outlier.clone().detach()

    inlier, outlier, targets_concat = prepare_images(inlier, outlier, target_inlier, target_outlier, device)

    # Calculate loss
    criterion = OutlierExposureLoss()

    adv_images = outlier.clone().detach()

    # Starting at a uniformly random point
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps_oe, eps_oe)

    # project back to epsilon ball
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    # apply adversarial attack steps times
    for _ in range(steps_oe):
        images_concat = torch.cat((inlier, adv_images))

        outputs = model(images_concat)

        # Calculate loss
        loss = criterion(outputs, targets_concat)

        # Update adversarial images
        grad = torch.autograd.grad(loss, inlier,
                                   retain_graph=False, create_graph=False)[0]

        # update adversarial images
        adv_images = adv_images.detach() + alpha_oe * grad.sign()

        # project back to epsilon ball
        delta = torch.clamp(adv_images - outlier, min=-eps_oe, max=eps_oe)

        # update adversarial images
        adv_images = outlier + delta

    if log:
        print(f"eps: {eps_oe} | Norm:{torch.linalg.norm(adv_images.cpu() - outlier.cpu()) / len(adv_images)}")

    return adv_images

def mifgsm_softmax(model, inlier, outlier, target_inlier, target_outlier, eps_oe, steps_oe, alpha_oe, device="gpu",
                   decay=1.0, log=False):
    """
    Momentum Iterative FGSM attack with softmax loss
    Args:
        model: model to attack
        inlier: input data from in-distribution
        outlier: input data from out-of-distribution
        target_inlier: target data from in-distribution
        target_outlier: target data from out-of-distribution
        eps_oe: epsilon for the attack
        steps_oe: number of steps for the attack
        alpha_oe: step size for the attack
        device: device to use
        decay: momentum decay
    Returns:
        adv_images: adversarial images
    """
    if eps_oe == 0:
        return outlier.clone().detach()

    inlier, outlier, targets_concat = prepare_images(inlier, outlier, target_inlier, target_outlier, device)

    # init empty momentum
    momentum = torch.zeros_like(inlier).detach().to(device)

    criterion = OutlierExposureLoss()

    adv_images = outlier.clone().detach()

    # Starting at a uniformly random point
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps_oe, eps_oe)

    # project back to epsilon ball
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    # apply adversarial attack steps times
    for _ in range(steps_oe):
        images_concat = torch.cat((inlier, adv_images))
        outputs = model(images_concat)

        # Calculate loss
        loss = criterion(outputs, targets_concat)

        # Update adversarial images
        grad = torch.autograd.grad(loss, inlier,
                                   retain_graph=False, create_graph=False)[0]

        # normalize gradient
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)

        # update gradient with momentum
        grad = grad + momentum * decay

        # update momentum
        momentum = grad

        # update adversarial images
        adv_images = adv_images.detach() + alpha_oe * grad.sign()

        # project back to epsilon ball
        delta = torch.clamp(adv_images - outlier, min=-eps_oe, max=eps_oe)

        # update adversarial images
        adv_images = outlier + delta

    if log:
        print(f"eps: {eps_oe} | Norm:{torch.linalg.norm(adv_images.cpu() - outlier.cpu()) / len(adv_images)}")

    return adv_images

def generate_adversarial_outlier(args, model, inlier, outlier, target_inlier, target_outlier, eps_oe, device):

    if eps_oe == 0:
        return outlier

    attack_methods = {
        "None": lambda *args, **kwargs: outlier,
        "FGSM": fgsm_softmax,
        "PGD": pgd_softmax,
        "MIFGSM": mifgsm_softmax,
        "EBA": energy_attack,
        "EPS_GAUSS": epsilon_gaussian_noise
    }

    if args.adv_oe in attack_methods:
        attack_method = attack_methods[args.adv_oe]
        common_kwargs = {
            "model": model,
            "inlier": inlier,
            "outlier": outlier,
            "target_inlier": target_inlier,
            "target_outlier": target_outlier,
            "eps_oe": eps_oe,
            "device": device
        }
        if args.adv_oe in ["PGD", "MIFGSM"]:
            common_kwargs.update({
                "alpha_oe": args.alpha_oe,
                "steps_oe": args.steps_oe
            })
        return attack_method(**common_kwargs)
    else:
        raise ValueError(f"Unsupported adversarial attack: {args.adv_oe}")
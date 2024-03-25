import numpy as np
import torch


def gaussian_noise(image_tensor, mean=0, var=0.1):
    """
    expects a tensor as image input
    :param mean:
    :param var:
    :return:
    """
    # Add Gaussian noise to the image
    sigma = var ** 0.5
    noisy_list = []
    gaussian = np.random.normal(mean, sigma, (
        image_tensor[0].shape[0], image_tensor[0].shape[1], image_tensor[0].shape[2]))
    gaussian = gaussian.reshape(image_tensor[0].shape[0], image_tensor[0].shape[1], image_tensor[0].shape[2])

    for image in image_tensor:
        noisy_image = image + gaussian
        noisy_list.append(noisy_image)

    noisy_tensor = torch.stack(noisy_list)

    return noisy_tensor


def salt_pepper(image_tensor, p):
    """
    expects a tensor as image input
    :param image_tensor:
    :param p: probability that a pixel gets turned to black or white
    :return:
    """

    mask = np.random.choice([0, 1, 2], size=image_tensor[0].shape[2:], p=[p / 2, p / 2, 1 - p])
    for i in range(len(image_tensor)):
        image_tensor[i][0][mask == 0] = -1
        image_tensor[i][1][mask == 0] = -1
        image_tensor[i][2][mask == 0] = -1
        image_tensor[i][0][mask == 1] = 1
        image_tensor[i][1][mask == 1] = 1
        image_tensor[i][2][mask == 1] = 1

    return image_tensor


def moire(image_tensor, wavelength, amplitude):
    """
    expects a tensor as image input
    :param wavelength:
    :param amplitude:
    :param image_tensor:
    :return:
    """

    # The wavelength of the pattern
    angle1 = 45  # The angle of the first pattern (in degrees)
    angle2 = 30  # The angle of the second pattern (in degrees)
    # The amplitude of the pattern

    # Create the moire pattern
    x = np.arange(0, image_tensor.shape[2], 1)
    y = np.arange(0, image_tensor.shape[3], 1)
    xx, yy = np.meshgrid(x, y)
    pattern1 = amplitude * np.sin(
        2 * np.pi / wavelength * (np.cos(angle1 * np.pi / 180) * xx + np.sin(angle1 * np.pi / 180) * yy))
    pattern2 = amplitude * np.sin(
        2 * np.pi / wavelength * (np.cos(angle2 * np.pi / 180) * xx + np.sin(angle2 * np.pi / 180) * yy))
    moire = pattern1 + pattern2

    # Add the moire pattern to the images
    images_with_moire = np.clip(image_tensor + moire, -1, 1)

    return images_with_moire

# Adversarial-Outlier-Exposure
This is the repository for the Workshop Paper: "Robust Out-of-Distribution Detection with Adversarial Outlier Exposure" by Konstantin Kirchheim and Thomas Botschen of OVGU Magdeburg.

Our work is based on the paper of https://arxiv.org/pdf/1812.04606.pdf and the github repository https://github.com/hendrycks/outlier-exposure.

Our new contribution is the integration of adversarial attacks on outliers during training.

## Installation
# TODO create requirements.txt

# TODO provide download links for the GAN-generated Outliers

In particular, this repository enables the user to do the following things:
- Finetune a pretrained WideResNet with (Adversarial) Outlier Exposure
- The Finetuning can get assisted by Adversarial Attacks on the Inliers and Outliers
- As Inlier Sets we currently support: CIFAR10, CIFAR100, ImageNet
- As Outlier Sets we use GAN-generated Images, which need to be manually created and added to the data folder /datasets
- Supported Adversarial Attacks: FGSM, PGD, MI-FGSM, EBA, EPS_GAUSS, None
---
Folder structure:
- config: default experiment configuration (will be overwritten by hydra)
- logic: 
    - eval.py: evaluation logic
    - training.py: training logic
    - utils.py: utility functions
    - attacks.py: adversarial attack logic

The training with default parameters can be run by:
>python3 run.py --config-name NAME_OF_CONFIG

e.g.
>python3 run.py --config-name cifar_adversarial_auroc.yaml

Results get saved in a .xlsx file called "results.xlsx".

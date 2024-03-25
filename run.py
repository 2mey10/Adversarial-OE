import sys
import os
import platform
import torch

def print_system_info():
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.path}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"PyTorch cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Current PyTorch Device: {torch.cuda.current_device()}")
    print(f"PyTorch Device Count: {torch.cuda.device_count()}")
    print(f"PyTorch Device Name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Device Capability: {torch.cuda.get_device_capability(0)}")



import hydra
from omegaconf import DictConfig
from logic.training import run


@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    print_system_info()
    my_app()

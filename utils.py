import torch
from torchsummary import summary

def get_device():
    """Gets the Device Type available on the machine"""
    if torch.cuda.is_available():
        print(f"Device Type - cuda")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print(f"Device Type - mps")
        return torch.device("mps")
    else:
        print(f"Device Type - cpu")
        return torch.device("cpu")


def print_model_summary(model, input_size = (3, 32, 32)):
    """Prints the model summary"""
    summary(model, input_size)


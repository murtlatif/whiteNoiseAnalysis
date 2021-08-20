import torch


def has_cuda():
    return torch.cuda.is_available()


def get_device():
    'cuda' if has_cuda() else 'cpu'

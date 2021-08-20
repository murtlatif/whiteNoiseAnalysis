import torch.utils.data
from torchvision import datasets, transforms


def fetch_mnist(batch_size: int, test_batch_size: int, use_cuda: bool):
    mnist_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mnist_train_data = datasets.MNIST('./data/mnistDataFiles', train=True, transform=mnist_transform, download=True)
    mnist_test_data = datasets.MNIST('./data/mnistDataFiles', train=False, transform=mnist_transform, download=True)

    train_kwargs = {
        'batch_size': batch_size,
        'num_workers': 1
    }
    test_kwargs = {
        'batch_size': test_batch_size,
        'num_workers': 1
    }

    if use_cuda:
        cuda_kwargs = {
            'num_workers': 4,
            'shuffle': True,
            'pin_memory': True
        }

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    mnist_train_loader = torch.utils.data.DataLoader(mnist_train_data, **train_kwargs)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test_data, **test_kwargs)

    return mnist_train_loader, mnist_test_loader

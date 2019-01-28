import torch

from torchvision.datasets import MNIST
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Lambda, Resize
from random import shuffle, sample

def int_to_one_hot(x, num_class):
    x.resize_(1, 1)
    x_onehot = torch.zeros(1, num_class)
    x_onehot.scatter_(1, x, 1)

    return x_onehot.view(num_class)


def one_hot_to_int(x_onehot, num_class):
    x = (x_onehot == 1).nonzero()
    return int(x[0, 0])


def load_MNIST(path=None, vectorize=True, target_onehot=True,
               num_train=-1, num_val=1):
    if path is None:
        path = "./mnist"

    # Transformations
    data_transform = [ToTensor()]
    target_transform = []
    if vectorize:
        data_transform.append(Lambda(lambda x: x.resize(28 * 28)))
    if target_onehot:
        target_transform.append(Lambda(lambda x: int_to_one_hot(x, 10)))

    # Loading dataset
    dataset = MNIST(path, train=True, download=True)
    if (num_train) > len(dataset) or num_train < 0:
        num_train = len(dataset)

    dataset.transform = Compose(data_transform)
    dataset.target_transform = Compose(target_transform)
    
    # Shuffling the dataset
    idx = sample(range(len(dataset)), num_train)
    train_dataset = Subset(dataset, idx)
    
    # Loading test dataset
    testset = MNIST(path, train=False, download=True)
    if num_val > len(testset):
        num_val = len(testset)

    testset.transform = Compose(data_transform)
    testset.target_transform = Compose(target_transform)
    
    # Shuffling the test dataset
    idx = sample(range(len(testset)), num_val)
    val_dataset = Subset(testset, idx)

    return train_dataset, val_dataset

from torchvision.datasets import MNIST
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Lambda
from random import shuffle, sample

def load_MNIST(path=None, num_train=-1, num_val=1, numbers=-1):
    if path is None:
        path = "./mnist"

    dataset = MNIST(path, train=True, download=True)
    if (num_train) > len(dataset) or num_train < 0:
        num_train = len(dataset)

    dataset.transform = Compose([ToTensor(), Lambda(lambda x: x.resize(28 * 28))])
    
    idx = sample(range(len(dataset)), num_train)
    train_dataset = Subset(dataset, idx)
    
    testset = MNIST("testdata/mnist", train=False, download=True)
    if num_val > len(testset):
        num_val = len(testset)
    
    testset.transform = Compose([ToTensor(), Lambda(lambda x: x.resize(28 * 28))])
    
    idx = sample(range(len(testset)), num_val)
    val_dataset = Subset(testset, idx)

    return train_dataset, val_dataset

import pandas as pd
from torchvision import transforms

from torch.utils.data import DataLoader, Subset, Dataset



class FlyDataset(Dataset):
    def __init__(self, path, mode='train'):
        self.imsize = (224, 224)
        self.transform = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])



    def __getitem__(self, idx):
        pass
        

def create_fly_datasets():

    big_train_dataset = FlyDataset(
        path='classification-of-butterflies/train_butterflies/train_split',
        mode='train'
    )
    pass
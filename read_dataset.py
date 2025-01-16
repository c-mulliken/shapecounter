import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import os

class ImageDataset(Dataset):
    def __init__(self, data_folder, label_file,
                 transform=transforms.ToTensor()):
        self.data_folder = data_folder
        self.label_file = label_file
        self.transform = transform

        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.image_filenames = list(self.labels.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.data_folder, f'{image_filename}.png')
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label_tuple = self.labels[image_filename]
        label = 0 if label_tuple[0] > label_tuple[1] else 1
        label = torch.tensor(label)

        return image, label
    
class CountDataset(Dataset):
    def __init__(self, data_folder, label_file,
                 transform=transforms.ToTensor()):
        self.data_folder = data_folder
        self.label_file = label_file
        self.transform = transform

        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.image_filenames = list(self.labels.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.data_folder, f'{image_filename}.png')
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label_tuple = self.labels[image_filename]
        label = torch.tensor(label_tuple)

        return image, label

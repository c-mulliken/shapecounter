import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from generate_dataset import create_dataset
from read_dataset import ImageDataset
from model import ResNet_18

# HYPERPARAMETERS
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = '/home/coby/Repositories/shapecounter/dataset/density_experiment'

# CREATE DATASETS
num_shape_means = np.arange(10, 100, 10)
num_shape_std = 3
mean_prop_circles = 0.5
std_prop_circles = 0.25
for mean in num_shape_means:
    create_dataset(10000, mean, num_shape_std, mean_prop_circles, std_prop_circles,
                   f'{PATH}/dataset_{mean}')
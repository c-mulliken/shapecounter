import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from read_dataset import ImageDataset
from model import ResNet_18

# HYPERPARAMETERS
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD DATASET
dataset = ImageDataset(data_folder='dataset/images', label_file='dataset/image_data.json',
                       transform=transforms.Compose([transforms.ToTensor()]))
trainset, testset = torch.utils.data.random_split(dataset, [4000, 1000])
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# LOAD MODEL
# model = ResNet_18(image_channels=1, num_classes=2).to(DEVICE)

# TRAINING LOOP
def train(model, train_loader, num_epochs, criterion, device):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 10 == 0:
                print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')

# torch.save(model, '/home/coby/Repositories/shapecounter/models/model.pt')
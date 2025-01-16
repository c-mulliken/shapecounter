import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from read_dataset import ImageDataset
from model import ResNet_18
import json
import sys

# SHELL ARGUMENT PARSING
MEAN = sys.argv[1]

# HYPERPARAMETERS
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = f'dataset/density_experiment/dataset_{MEAN}'

# LOAD DATASET
dataset = ImageDataset(data_folder=PATH,
                       label_file=f'{PATH}/image_data.json',
                       transform=transforms.Compose([transforms.ToTensor()]))
trainset, testset = torch.utils.data.Subset(dataset, range(8000)), torch.utils.data.Subset(dataset, range(8000, 10000))
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# LOAD MODEL
model = ResNet_18(image_channels=1, num_classes=2).to(DEVICE)

# TRAINING LOOP
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    for i, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 10 == 0:
            print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')

# SAVE MODEL
torch.save(model, f'dataset/density_experiment/model_{MEAN}.pt')

# SAVE LOSSES
with open('dataset/density_experiment/losses.json', 'r') as f:
    loss_dict = json.load(f)

loss_dict[MEAN] = losses

with open('dataset/density_experiment/losses.json', 'w') as f:
    json.dump(loss_dict, f)

# torch.save(model, '/home/coby/Repositories/shapecounter/models/model.pt')
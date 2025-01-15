import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from read_dataset import ImageDataset
from model import ResNet_18

# HYPERPARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# LOAD DATASET
dataset = ImageDataset(data_folder='dataset/images', label_file='dataset/image_data.json',
                       transform=transforms.Compose([transforms.ToTensor()]))
trainset, testset = torch.utils.data.random_split(dataset, [80, 20])
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# LOAD MODEL
model = ResNet_18(image_channels=1, num_classes=2)

# TRAINING LOOP
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ResNet_18

def head(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    
    return x

class StitchedModel(nn.Module):
    def __init__(self, model_head):
        super().__init__()
        self.model_head = model_head

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = head(self.model_head, x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x
# Facial Emotion Recognition Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module)
    def __init__(self, channels, reduction=16)
        super().__init__()
        self.fc1 = nn.Linear(channels, channels  reduction)
        self.fc2 = nn.Linear(channels  reduction, channels)

    def forward(self, x)
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x  y.expand_as(x)

class A_LightCNN(nn.Module)
    def __init__(self, num_classes=7)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.se1 = SEBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.se2 = SEBlock(64)
        self.fc = nn.Linear(64  28  28, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x)
        x = F.relu(self.se1(self.conv1(x)))
        x = F.relu(self.se2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        out = self.classifier(embedding)
        return out, embedding
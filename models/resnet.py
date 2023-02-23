import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = identity + out.clone()     
        
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128, 128)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResBlock(512, 512)
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.maxpool(out)
        # Flatten the output and pass it through the fully connected layer
        out = out.view(out.size(0), -1)
        out = self.fc(out)        
        # Apply the softmax activation function to the output
        #out = self.softmax(out)
        #print('output')
        out = F.log_softmax(out)
        return out
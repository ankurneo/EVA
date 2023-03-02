import torch
import torch.nn as nn
import torch.nn.functional as F

class UltimusBlock(nn.Module):
    def __init__(self, in_features, out_features, k_features):
        super(UltimusBlock, self).__init__()
        self.fc_k = nn.Linear(in_features, k_features)
        self.fc_q = nn.Linear(in_features, k_features)
        self.fc_v = nn.Linear(in_features, k_features)
        self.fc_out = nn.Linear(k_features, out_features)
        
    def forward(self, x):
        k = self.fc_k(x)
        q = self.fc_q(x)
        v = self.fc_v(x)
        am = nn.functional.softmax(torch.matmul(q.t(), k) / (8 ** 0.5), dim=-1)
        z = torch.matmul(v,am)
        return self.fc_out(z)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ultimus1 = UltimusBlock(48, 48, 8)
        self.ultimus2 = UltimusBlock(48, 48, 8)
        self.ultimus3 = UltimusBlock(48, 48, 8)
        self.ultimus4 = UltimusBlock(48, 48, 8)
        self.fc_out = nn.Linear(48, 10)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.ultimus1(x)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)
        x = self.fc_out(x)
        return x
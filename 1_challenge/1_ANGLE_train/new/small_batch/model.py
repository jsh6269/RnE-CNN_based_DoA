import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class DoANet(nn.Module):
    def __init__(self):
        super().__init__()
        nclass = 4
        self.cbl1 = nn.Sequential(
            nn.Conv1d(160, 80, 5),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(0.3)
        )
        self.fc1 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.cbl1(x)
        x, _ = x.max(dim=-1)
        #x = x.mean(dim=-1)
        x = self.fc1(x)

        return x

if __name__=='__main__':
    net = DoANet()


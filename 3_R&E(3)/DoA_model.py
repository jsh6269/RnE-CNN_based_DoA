import torch.nn as nn

class DoANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cbl1 = nn.Sequential(
            nn.Conv2d(2, 128, 3, padding=1),
            nn.AvgPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.AvgPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.25)
        )

        self.cbl2 = nn.Sequential(
            nn.Conv1d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.cbl1(x)
        x = x.view(x.shape[0], 128, -1)
        x = self.cbl2(x)
        x, _ = x.max(dim=-1)
        x = self.fc1(x)
        return x

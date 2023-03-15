import torch.nn as nn


class VADNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cbl1 = nn.Sequential(
            nn.Conv2d(2, 72, 3, padding=1),
            nn.AvgPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(72),
            nn.Dropout(0.1),
            nn.Conv2d(72, 144, 3, padding=1),
            nn.AvgPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(144),
            nn.Dropout(0.2)
        )

        self.cbl2 = nn.Sequential(
            nn.Conv1d(72, 144, 3),
            nn.ReLU(),
            nn.BatchNorm1d(144)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(144, 72),
            nn.ReLU(),
            nn.BatchNorm1d(72),
            nn.Dropout(0.2),
            nn.Linear(72, 2)
        )

    def forward(self, x):
        x = self.cbl1(x)
        x = x.view(x.shape[0], 72, -1)
        x = self.cbl2(x)
        x, _ = x.max(dim=-1)
        x = self.fc1(x)
        return x

import torch.nn as nn

class DoANet(nn.Module):
    def __init__(self, input_channel):
        super().__init__()

        self.cbl1 = nn.Sequential(
            nn.Conv2d(input_channel, 48, 5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(48, 24, 5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.cbl2 = nn.Sequential(
            nn.Conv1d(24 * 20, 48, 5),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(48, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 4)
        )

    def forward(self, x):
        x = self.cbl1(x)
        x = x.view(x.shape[0], 24 * 20, -1)
        x = self.cbl2(x)
        x, _ = x.max(dim=-1)
        x = self.fc1(x)
        return x

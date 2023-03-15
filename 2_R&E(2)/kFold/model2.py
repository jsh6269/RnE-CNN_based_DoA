import torch.nn as nn

class DoANet(nn.Module):
    def __init__(self, input_channel=2):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv1d(160, 120, 5),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Dropout(0.25)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(120, 160),
            nn.ReLU(),
            nn.BatchNorm1d(160),
            nn.Dropout(0.5),
            nn.Linear(160, 10)
    )
    def forward(self, x):
        x = self.cbl1(x)
        # x, _ = x.max(dim=-1)
        x = x.mean(dim=-1)
        x = self.fc1(x)

        return x

if __name__=='__main__':
    net = DoANet()


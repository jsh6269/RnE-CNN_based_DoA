# File Tree
# .
# ├── code.py
# ├── Data
# │   ├── 00001.wav
# │   ├── 00002.wav
# │   ├── 00003.wav
# │   ├── 00004.wav
# │   └── 00005.wav
# └── wavft.npy

import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
from glob import glob
import librosa
import soundfile as sf
from matplotlib import pyplot as plt

sys.stdout = open('./log.txt', 'w')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

wavlist = np.sort(glob('./Data/*.wav'))

wavft = []
wavgt = []

for i in range(10):
    for _ in range(116):
        wavgt.append(int(i))

# 전처리, Noise 제거 안 함
def logmel(x, sr, mel_dim=64):
    hop = int(64 / 1000 * sr)
    win = int(128 / 1000 * sr)
    X0 = librosa.feature.melspectrogram(y=x[:, 0], sr=sr, n_mels=mel_dim, hop_length=hop, n_fft=win)
    X1 = librosa.feature.melspectrogram(y=x[:, 1], sr=sr, n_mels=mel_dim, hop_length=hop, n_fft=win)
    X0 = librosa.power_to_db(X0)
    X1 = librosa.power_to_db(X1)
    X = np.vstack((X0, X1))
    return X

for wav in wavlist:
    x, sr = sf.read(wav)
    wavft.append(logmel(x, sr))

np.save('./wavft', wavft)
wavft = np.load('./wavft.npy')

X = torch.tensor(wavft, dtype=torch.float32)
Y = torch.tensor(wavgt, dtype=torch.int64)
kf = StratifiedKFold(n_splits=5, shuffle=True)

class DoANet(nn.Module):
    def __init__(self):
        super().__init__()
        nclass = 4
        self.cbl1 = nn.Sequential(
            nn.Conv1d(128, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(16, 10)
    def forward(self, x):
        x = self.cbl1(x)
        x, _ = x.max(dim=-1)
        x = self.fc1(x)
        return x

fold = 0
for train_idx, test_idx in kf.split(X,Y):
    fold = fold + 1
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    ds = TensorDataset(X_train, Y_train)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    net = DoANet()
    net = net.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0025)

    train_losses = []
    test_losses = []
    test_accs = []

    for epoch in range(100):
        net.train()
        running_loss = 0.
        for i, (xx, yy) in enumerate(loader):
            xx, yy = xx.cuda(), yy.cuda()
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss/i)

        net.eval()
        X_test, Y_test = X_test.cuda(), Y_test.cuda()
        y_pred = net(X_test)
        test_loss = loss_fn(y_pred, Y_test)
        test_losses.append(test_loss.item())
        with torch.no_grad():
            _, pred = y_pred.max(1)
        test_acc = (Y_test == pred).sum()
        test_accs.append(test_acc.item()/len(pred))
        print('Epoch :', '(' + str(fold) + ')', '%05d' % epoch, '  Train_loss: ', '%.6f' % train_losses[-1],
              '  Valid_loss :', '%.6f' % test_losses[-1], '  Valid_acc: ', '%.6f' % test_accs[-1])

    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 5)
    ax1.set(xlim=(0, 100), ylim=(-0.02, 2.02))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(train_losses, label='Train Loss', color='mediumslateblue')
    ax1.plot(test_losses, label='Valid Loss', color='slateblue')
    ax1.tick_params(axis='y', labelcolor='darkslateblue')
    plt.legend(loc=7)
    ax2 = ax1.twinx()
    ax2.set(ylim=(-0.01, 1.01))
    ax2.set_ylabel('Accuracy')
    ax2.plot(test_accs, label='Valid Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    plt.title("k-Fold Validation : " + str(fold) + " Fold")
    fig.savefig('./' + str(fold) + '_Fold', dpi=600)

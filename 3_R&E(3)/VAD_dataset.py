import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib
import torch, random
import torch.utils.data as data
import numpy as np
from Tool_kit import commander

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Dataset(data.Dataset):
    def __init__(self, which_set, kindex, kfold, npPath, gtPath, gpu):
        # npPath = ['./rneData/32XAudioNoise/bees.npy', ... , './rneData/32XAudioNoise/treeWind.npy']
        # gtPath = './rneData/vad.npy'
        self.npPath = npPath
        self.diverse = len(npPath)  # 70
        self.melSize = (1160, 2, 80, 244)
        self.splitNum = 39
        self.gtData = np.load(gtPath)  # (1160, 39)

        np.random.seed(1)
        perm = np.random.permutation(np.arange(self.melSize[0]))
        perm_k_block = perm.reshape((kfold, -1))

        valid_idx = perm_k_block[kindex]
        train_idx = np.asarray([xx for xx in perm if xx not in valid_idx])
        test_idx = perm

        self.IDs = []
        if which_set == 'test':
            self.idx = test_idx
        else:
            self.idx = train_idx if which_set == 'train' else valid_idx

        # diverse, idx, start, finish, gt
        for diverse in range(self.diverse):
            for dataNum in self.idx:
                for i in range(self.splitNum):
                    start = int(i * (self.melSize[3] / self.splitNum))
                    finish = start + 6
                    # diverse, dataNum, start, finish, gt, intervalNum
                    self.IDs.append((diverse, dataNum, start, finish, torch.tensor(self.gtData[dataNum][i], dtype=torch.int64), i))

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        # diverse, idx, start, finish, gt
        folder = self.npPath[ID[0]]
        x = folder+'/'+str(ID[1]).zfill(4)
        x = torch.load(x)
        x = x[:, :, ID[2]:ID[3]]
        gt = ID[4]
        return x, gt, self.IDs[index]


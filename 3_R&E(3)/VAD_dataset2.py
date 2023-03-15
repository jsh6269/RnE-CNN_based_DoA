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
        self.melSize = np.load(npPath[0]).shape  # (1160, 2, 80, 244)
        self.splitNum = 39
        self.gtData = np.load(gtPath)  # (1160, 39)
        self.which_set = which_set

        np.random.seed(1)
        perm = np.random.permutation(np.arange(self.melSize[0]))
        perm_k_block = perm.reshape((kfold, -1))

        valid_idx = perm_k_block[kindex]
        train_idx = np.asarray([xx for xx in perm if xx not in valid_idx])

        if which_set == 'valid':
            self.IDs = []
            # 0, valid_idx, :, start, finish, gt
            for dataNum in valid_idx:
                for i in range(self.splitNum):
                    start = int(i * (self.melSize[3] / self.splitNum))
                    finish = start + 6
                    self.valid_ID.append((0, dataNum, -1, start, finish, self.gtData[dataNum][i]))

        if which_set == 'train':
            self.train_ID = []
            # diverse, train_idx, :, start, finish
            for diverse in range(self.diverse):
                for dataNum in train_idx:
                    for i in range(self.splitNum):
                        start = int(i * (self.melSize[3] / self.splitNum))
                        finish = start + 6
                        self.train_ID.append((diverse, dataNum, -1, start, finish, self.gtData[dataNum][i]))


    def __len__(self):
        if self.which_set == 'valid':
            return len(self.valid_ID)
        elif self.which_set == 'train':
            return len(self.train_ID)

    def __getitem__(self, index):
        x = self.data[index]
        # x = torch.cat((x, x, x, x), dim=2)
        gt = self.gt[index]
        # x ; [39, 2, 80, 6]
        return x, gt


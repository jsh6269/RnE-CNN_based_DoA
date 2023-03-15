import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import matplotlib
import torch, random
import torch.utils.data as data
import numpy as np
matplotlib.use('Agg')

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Dataset(data.Dataset):
    def __init__(self, which_set, kindex, kfold, npPath, gpu, vadPt):
        # npPath = ['./rneData/32XAudioNoise/bees.npy', ... , './rneData/32XAudioNoise/treeWind.npy']
        self.npData = np.asarray([np.load(path) for path in npPath])
        # self.npData ; [?, 1160, 2, 80, 244]
        self.gtData = np.asarray([int(i/116) for i in range(1160)])
        self.numData = np.asarray([i for i in range(1160)])
        # self.gtData ; [1160]
        self.dataNum = self.npData.shape[1]
        # self.dataNum = 1160
        self.diverse = self.npData.shape[0]
        self.gpu = gpu

        np.random.seed(1)
        perm = np.random.permutation(np.arange(self.dataNum))
        perm_k_block = perm.reshape((kfold, -1))

        valid_idx = perm_k_block[kindex]
        train_idx = np.asarray([xx for xx in perm if xx not in valid_idx])

        if which_set == 'train':
            # noinspection All
            self.melData = torch.tensor(np.concatenate([self.npData[i][train_idx] for i in range(self.diverse)]), dtype=torch.float32)
            # noinspection All
            self.gtData = torch.tensor(np.concatenate([self.gtData[train_idx] for i in range(self.diverse)]), dtype=torch.int64)
            self.numlist = [self.numData[train_idx] for i in range(self.diverse)][0]

        elif which_set == 'valid':
            # noinspection All
            self.melData = torch.tensor(self.npData[0][valid_idx], dtype=torch.float32)
            # noinspection All
            self.gtData = torch.tensor(self.gtData[valid_idx], dtype=torch.int64)
            self.numlist = [self.numData[train_idx] for i in range(self.diverse)][0]

        if self.gpu:
            self.melData = self.melData.cuda()
            self.gtData = self.gtData.cuda()

    def __len__(self):
        assert len(self.melData) == len(self.gtData)
        return len(self.melData)

    def __getitem__(self, index):
        x = self.melData[index]
        gt = self.gtData[index]
        id = self.numlist[index]
        return x, gt, id

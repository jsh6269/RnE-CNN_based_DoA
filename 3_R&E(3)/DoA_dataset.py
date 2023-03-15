import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
import torch, random
import torch.utils.data as data
import numpy as np
matplotlib.use('Agg')
from VAD_model import VADNet
from Tool_kit import commander

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Dataset(data.Dataset):
    def __init__(self, which_set, kindex, kfold, npPath, gpu, vadPred, mode='slice'):
        # npPath = ['./rneData/32XAudioNoise/bees.npy', ... , './rneData/32XAudioNoise/treeWind.npy']

        self.npPath = npPath
        self.diverse = len(npPath)  # 70
        self.melSize = (1160, 2, 80, 244)
        self.splitNum = 39
        self.dataNum = self.melSize[0]
        self.diverse = len(self.npPath)
        self.gpu = gpu
        self.mode = mode

        np.random.seed(1)
        perm = np.random.permutation(np.arange(self.melSize[0]))
        perm_k_block = perm.reshape((kfold, -1))

        valid_idx = perm_k_block[kindex]
        train_idx = np.asarray([xx for xx in perm if xx not in valid_idx])

        self.IDs = []
        self.idx = train_idx if which_set == 'train' else valid_idx
        # diverse, idx, start, finish, gt

        if self.mode != 'origin':
            for diverse in range(self.diverse):
                for dataNum in self.idx:
                    for i in range(self.splitNum):
                        start = int(i * (self.melSize[3] / self.splitNum))
                        finish = start + 6
                        if which_set == "valid":
                            isSpeech = vadPred[diverse][dataNum][i]
                        elif which_set == "train":
                            isSpeech = vadPred[diverse+1][dataNum][i]
                        else:
                            raise AssertionError
                        # shape of vadPred : (diverse, 1160, 39)
                        # We should make vadPred (the source file which make vadPred (which is an estimation of trained VAD_model))
                        # We prolong the procedure since there's not much time
                        if isSpeech:
                            self.IDs.append((diverse, dataNum, start, finish, torch.tensor(dataNum/116, dtype=torch.int64)))

        if self.mode == 'origin':
            for diverse in range(self.diverse):
                for dataNum in self.idx:
                    self.IDs.append((diverse, dataNum, 0, 0.1, torch.tensor(dataNum/116, dtype=torch.int64)))


        # self.gtData = np.asarray([int(i/116) for i in range(1160)])
        #
        #
        # self.npData = np.asarray([np.load(path) for path in npPath])
        # # self.npData ; [?, 1160, 2, 80, 244]
        # self.gtData = np.asarray([int(i/116) for i in range(1160)])
        # self.numData = np.asarray([i for i in range(1160)])
        # # self.gtData ; [1160]
        # self.dataNum = self.npData.shape[1]
        # # self.dataNum = 1160
        # self.diverse = self.npData.shape[0]
        #
        # np.random.seed(1)
        # perm = np.random.permutation(np.arange(self.dataNum))
        # perm_k_block = perm.reshape((kfold, -1))
        #
        # valid_idx = perm_k_block[kindex]
        # train_idx = np.asarray([xx for xx in perm if xx not in valid_idx])
        #
        # if which_set == 'train':
        #     # noinspection All
        #     self.melData = torch.tensor(np.concatenate([self.npData[i][train_idx] for i in range(self.diverse)]), dtype=torch.float32)
        #     # noinspection All
        #     self.gtlist = torch.tensor(np.concatenate([self.gtData[train_idx] for i in range(self.diverse)]), dtype=torch.int64)
        #     self.numlist = [self.numData[train_idx] for i in range(self.diverse)][0]
        #
        # elif which_set == 'valid':
        #     # noinspection All
        #     self.melData = torch.tensor(self.npData[0][valid_idx], dtype=torch.float32)
        #     # noinspection All
        #     self.gtlist = torch.tensor(self.gtData[valid_idx], dtype=torch.int64)
        #     self.numlist = [self.numData[train_idx] for i in range(self.diverse)][0]
        #
        # self.melData = commander(self.melData)
        #
        # if self.gpu:
        #     self.melData = self.melData.cuda()
        #
        # self.data, self.gt, self.id = [], [], []
        # for i in range(self.melData.shape[0]):
        #     vadGap = torch.argmax(vad(self.melData[i]), dim=1)
        #     for j in range(self.melData.shape[1]):
        #         if vadGap[j] == 1:
        #             self.gt.append(int(self.gtlist[i]))
        #             self.data.append(self.melData[i][j].tolist())
        #             self.id.append(self.numlist[i])
        #
        # # noinspection All
        # self.data = torch.tensor(self.data, dtype=torch.float32)
        # # noinspection All
        # self.gt = torch.tensor(self.gt, dtype=torch.int64)
        #
        # if self.gpu:
        #     self.data = self.data.cuda()
        #     self.gt = self.gt.cuda()

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        # diverse, idx, start, finish, gt
        folder = self.npPath[ID[0]]
        x = folder+'/'+str(ID[1]).zfill(4)
        x = torch.load(x)
        if self.mode != 'origin':
            x = x[:, :, ID[2]:ID[3]]
        gt = ID[4]
        return x, gt, ID[1]


        # x = self.data[index]
        # gt = self.gt[index]
        # id = self.id[index]
        # return x, gt, id

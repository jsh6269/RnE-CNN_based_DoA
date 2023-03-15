import matplotlib
import torch, random
import torch.utils.data as data
from os.path import join
import os
from glob import glob
import numpy as np
matplotlib.use('Agg')

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Dataset(data.Dataset):
    def __init__(self, which_set, kindex, kfold, datapath):
        self.datapath = datapath
        self.which_set = which_set
        self.wavlist = np.sort(glob(join(self.datapath,'*.npy')))
        self.prelist = [xx for xx in self.wavlist if 'Aug' not in xx]

        np.random.seed(1)
        perm = np.random.permutation(self.prelist)
        perm_k_block = perm.reshape((kfold, -1))

        self.validlist = perm_k_block[kindex]
        self.trainlist = np.asarray([xx for xx in perm if xx not in self.validlist])

        for path in self.trainlist:
            prename = path.split('/')[-1]
            aug_azimuth = 180 - int(prename.split('_')[0])
            augname = str(aug_azimuth).zfill(3) + '_' + prename.split('_')[1] + '_Aug_wavft.npy'
            aug_path = self.datapath + '/' + augname
            if os.path.exists(aug_path):
                self.trainlist = np.append(self.trainlist, aug_path)

        if self.which_set == 'train':
            self.filelist = self.trainlist.copy()
        elif self.which_set == 'valid':
            self.filelist = self.validlist.copy()

        self.gtlist = self._read_gt()
        print("Loading numpy data for", self.which_set, 'in progress...')
        self.wavftlist = [np.load(xx) for xx in self.filelist]

        self.wavftlist = torch.tensor(self.wavftlist, dtype=torch.float32)
        self.gtlist = torch.tensor(self.gtlist, dtype=torch.int64)

    def __len__(self):
        return len(self.wavftlist)

    def __getitem__(self, index):
        wavpath = self.filelist[index]
        gt = self.gtlist[index]
        X = self.wavftlist[index]
        # X = np.vstack([X[0], X[1]])
        return X, gt, wavpath

    def _read_gt(self):
        conversion = {0: 0, 20: 1, 40: 2, 60: 3, 80: 4, 100: 5, 120: 6, 140: 7, 160: 8, 180: 9}
        convert_gt = lambda x:conversion[x]
        gt = []
        for filepath in self.filelist:
            file_name = filepath.split('/')[-1]
            gt.append(convert_gt(int(file_name.split('_')[0])))
        gt = np.asarray(gt)
        return gt

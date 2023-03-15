import matplotlib
import torch, json, random
import torch.utils.data as data
from os.path import join
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
    def __init__(self, which_set, kindex, kfold, datapath, order):
        self.datapath = datapath
        self.gtname = 't3_gt.json'
        self.wavlist = np.sort(glob(join(self.datapath, '*.npy')))
        self.gtlist = self._read_gt()
        assert len(self.gtlist) == len(self.wavlist)

        print("loading numpy data%d... please wait..." %order)

        self.wavft = []
        wavftlist = np.sort(glob(datapath+'/*.npy'))

        for npdata in wavftlist:
            self.wavft.append(np.load(npdata))

        self.wavft = torch.tensor(self.wavft, dtype=torch.float32)
        self.gtlist = torch.tensor(self.gtlist, dtype=torch.int64)

        if order==3: print("it takes a few sec to get started...")
        np.random.seed(1)
        perm = np.random.permutation(np.arange(len(self.wavlist)))
        perm_k_block = perm.reshape((kfold, -1))

        valid_idx = perm_k_block[kindex]
        train_idx = np.asarray([xx for xx in perm if xx not in valid_idx])

        if which_set == 'train':
            self.index = train_idx
            self.wavlist = self.wavlist[train_idx]
            self.gtlist = self.gtlist[train_idx]
        elif which_set == 'valid':
            self.index = valid_idx
            self.wavlist = self.wavlist[valid_idx]
            self.gtlist = self.gtlist[valid_idx]

    def __len__(self):
        assert len(self.gtlist) == len(self.wavlist)
        return len(self.wavlist)

    def __getitem__(self, index):
        wavpath = self.wavlist[index]
        gt = self.gtlist[index]
        X = self.wavft[self.index[index]]
        return X, gt, wavpath

    def _read_gt(self):
        conversion = {45: 0, 60: 1, 75: 2, 90: 3}
        convert_gt = lambda x:conversion[x]
        gt = []
        for filepath in self.wavlist:
            file_name = filepath.split('\\')[-1]
            gt.append(convert_gt(int(file_name[:2])))
        gt = np.asarray(gt)
        # gt = json.load(open(join(self.datapath, self.gtname), 'r'))['track3_results']
        # gt = np.asarray([convert_gt(xx['angle']) for xx in gt])
        return gt

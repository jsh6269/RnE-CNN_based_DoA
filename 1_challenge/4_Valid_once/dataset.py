import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import os, re, random
from os.path import join, splitext, basename, dirname
from glob import glob
import soundfile as sf
import numpy as np
import json, librosa
import librosa.display
from enhancement import ss, hpf

random.seed(123)
np.random.seed(123)

class Dataset(data.Dataset):
    def __init__(self, which_set='train', kindex=0, kfold=5):
        self.datapath = './Data'
        self.gtname = 't3_gt.json'
        self.wavlist = np.sort(glob(join(self.datapath, '*.wav')))
        self.gtlist = self._read_gt()

        assert len(self.gtlist) == len(self.wavlist)
        n_samples = len(self.gtlist)

    def __len__(self):
        assert len(self.gtlist) == len(self.wavlist)
        return len(self.wavlist)

    def __getitem__(self, index):
        wavpath = self.wavlist[index]
        x, sr = sf.read(wavpath)

        # Enhancement
        x = ss(x, sr)
        x = hpf(x, sr)

        X = self._logmel(x, sr)
        gt = self.gtlist[index]
        return X, gt, wavpath

    def _read_gt(self):
        #convert_gt = lambda x:x//20
        gt = json.load(open(join(self.datapath, self.gtname), 'r'))['track3_results']
        gt = np.asarray([xx['angle'] for xx in gt])
        return gt

    def _logmel(self, x, sr, mel_dim=80, hop=10, win=30):

        if x.shape[0] != 2 and x.shape[1] == 2:
            x = x.T 

        hop = int(10 / 1000 * sr)
        win = int(25 / 1000 * sr)

        X0 = librosa.feature.melspectrogram(y=x[0, :], sr=sr, n_mels=mel_dim, hop_length=hop, n_fft=win)
        X1 = librosa.feature.melspectrogram(y=x[1, :], sr=sr, n_mels=mel_dim, hop_length=hop, n_fft=win)

        X0 = librosa.power_to_db(X0)
        X1 = librosa.power_to_db(X1)

        #plt.figure()
        #librosa.display.specshow(X0, y_axis='mel', x_axis='time', fmax=8000, cmap='magma')
        #plt.savefig('aa.png')
        #plt.close()
        
        X = np.stack((X0, X1), axis=0)
        return X

def collate_fn(data):
    print(list(zip(*data)))
    datas, ids, fpath = list(zip(*data))

    ids = [int(xx) for xx in ids]
    min_len = np.min([xx.shape[-1] for xx in datas])
    
    dshape = datas[0].shape
    data_cuts = torch.zeros(len(datas), dshape[0] * dshape[1], min_len)

    for idx, data in enumerate(datas):
        data = data.reshape(data.shape[0] * data.shape[1], -1)
        data_cuts[idx] = torch.tensor(data[..., :min_len])

    return data_cuts, torch.tensor(ids), fpath

if __name__=='__main__':
    aa = Dataset()
    batch = [xx for xx in aa]
    batch = collate_fn(batch)
    import ipdb
    ipdb.set_trace()

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
    def __init__(self, which_set):
        self.datapath = './Data'
        self.wavlist = np.sort(glob(join(self.datapath, '*.wav')))

    def __len__(self):
        return len(self.wavlist)

    def __getitem__(self, index):
        wavpath = self.wavlist[index]
        x, sr = sf.read(wavpath)
        #x = x/np.amax(np.abs(x))
        # Enhancement
        x = ss(x, sr)
        x = hpf(x, sr)

        X = self._logmel(x, sr)
        return X, wavpath

    def _logmel(self, x, sr, mel_dim=80, hop=10, win=30):

        if x.shape[0] != 2 and x.shape[1] == 2:
            x = x.T 

        hop = int(10 / 1000 * sr)
        win = int(25 / 1000 * sr)

        X0 = librosa.feature.melspectrogram(y=x[0, :], sr=sr, n_mels=mel_dim, hop_length=hop, n_fft=win)
        X1 = librosa.feature.melspectrogram(y=x[1, :], sr=sr, n_mels=mel_dim, hop_length=hop, n_fft=win)

        X0 = librosa.power_to_db(X0)
        X1 = librosa.power_to_db(X1)

        X = np.stack((X0, X1), axis=0)
        return X

def collate_fn(data):
    datas, fpath = list(zip(*data))

    min_len = np.min([xx.shape[-1] for xx in datas])
    
    dshape = datas[0].shape
    data_cuts = torch.zeros(len(datas), dshape[0] * dshape[1], min_len)

    for idx, data in enumerate(datas):
        data = data.reshape(data.shape[0] * data.shape[1], -1)
        data_cuts[idx] = torch.tensor(data[..., :min_len])

    return data_cuts, fpath

if __name__=='__main__':
    aa = Dataset()
    batch = [xx for xx in aa]
    batch = collate_fn(batch)
    import ipdb
    ipdb.set_trace()

from glob import glob
import numpy as np
import librosa, os
from scipy import signal
import soundfile as sf

pathread = '/home/rne/Documents/DOA/RnE/Noise1Audio2/*.wav'

def hpf(x, sr, ch):
    sos = signal.butter(10, 250, 'hp', fs=sr, output='sos')
    x_filtered = np.zeros_like(x)
    for i in range(ch):
        x_filtered[:,i] = signal.sosfilt(sos, x[:,i])
    return x_filtered


def spectral_subtraction(x, sr, ch, st=None, end=None):
    st = 5 if st is None else st
    end = 5 if end is None else end

    hop = int(10 / 1000 * sr)
    win = int(25 / 1000 * sr)
    X=[]

    for i in range(ch):
        S = librosa.stft(x[:, i], hop_length=hop, win_length=win)
        S1_mag, S1_phase = np.abs(S), np.angle(S)

        S1_noise = np.sum(S1_mag[:, :st], axis=1, keepdims=True) + np.sum(S1_mag[:, -end:], axis=1, keepdims=True)
        S1_noise = S1_noise / (st + end)

        S1_mag2 = np.maximum(S1_mag - S1_noise, 0)

        S1_out = S1_mag2 * np.exp(S1_phase * 1j)

        x1_out = librosa.istft(S1_out, hop_length=hop, win_length=win)
        X.append(x1_out)

    x_out = np.stack(X, axis=1)

    return x_out

def logmel(x, sr, ch, mel_dim=120):

    if x.shape[0] != ch and x.shape[1] == ch:
        x = x.T

    hop = int(8 / 1000 * sr)
    win = int(32 / 1000 * sr)

    X = []
    for i in range(ch):
        Xi = librosa.feature.melspectrogram(y=x[i, : ], sr=sr, n_mels=mel_dim, hop_length=hop, n_fft=win)
        Xi = librosa.power_to_db(Xi)
        Xi = Xi[10:90, :]
        X.append(Xi)

    X = np.stack(X, axis=0)

    return X

filelist = np.sort(glob(pathread))

min = 99999999
result = []

for index in range(len(filelist)):
    ch = 2
    x, sr = sf.read(filelist[index])
    x = hpf(x, sr, ch)
    x = spectral_subtraction(x, sr, ch)
    X = logmel(x, sr, ch)

    if min > len(X[0,0,:]):
        min = len(X[0,0,:])

    result.append(X)
    print(index+1, "/", len(filelist), "\t%.3f%% completed" %((index+1)*100/len(filelist)))

folder = './wavft/Noise1Audio2/'
if not os.path.exists(folder):
    os.makedirs(folder)

for index in range(len(filelist)):
    X = result[index]
    X = X[:,:,:min]
    num = "%04d" %index
    filename = (filelist[index].split('/'))[-1]
    np.save(folder+filename[:-4]+'_wavft', X, allow_pickle=True)

print("min_size:", min)

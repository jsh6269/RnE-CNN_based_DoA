import librosa
from scipy import signal
import numpy as np

def hpf(x, sr):
    sos = signal.butter(10, 250, 'hp', fs=sr, output='sos')
    x_filtered = np.zeros_like(x)
    x_filtered[:,0] = signal.sosfilt(sos, x[:,0])
    x_filtered[:,1] = signal.sosfilt(sos, x[:,1])
    return x_filtered

def lpf(x, sr):
    sos = signal.butter(10, 250, 'lp', fs=sr, output='sos')
    x_filtered = np.zeros_like(x)
    x_filtered[:,0] = signal.sosfilt(sos, x[:,0])
    x_filtered[:,1] = signal.sosfilt(sos, x[:,1])
    return x_filtered

def spectral_subtraction(x, sr, st=None, end=None):
    # Estimate noise: Let assume first and last 5 frames...
    st = 5 if st is None else st
    end = 5 if end is None else end

    hop = int(10 / 1000 * sr)
    win = int(25 / 1000 * sr)

    S1 = librosa.stft(x[:,0], hop_length=hop, win_length=win)
    S2 = librosa.stft(x[:,1], hop_length=hop, win_length=win)

    S1_mag, S1_phase = np.abs(S1), np.angle(S1)
    S2_mag, S2_phase = np.abs(S2), np.angle(S2)

    S1_noise = np.sum(S1_mag[:, :st], axis=1, keepdims=True) + np.sum(S1_mag[:, -end:], axis=1, keepdims=True)
    S1_noise = S1_noise / (st + end)

    S2_noise = np.sum(S2_mag[:, :st], axis=1, keepdims=True) + np.sum(S2_mag[:, -end:], axis=1, keepdims=True)
    S2_noise = S2_noise / (st + end)

    S1_mag2 = np.maximum(S1_mag - S1_noise, 0)
    S2_mag2 = np.maximum(S2_mag - S2_noise, 0)

    S1_out = S1_mag2 * np.exp(S1_phase * 1j)
    S2_out = S2_mag2 * np.exp(S2_phase * 1j)

    x1_out = librosa.istft(S1_out, hop_length=hop, win_length=win)
    x2_out = librosa.istft(S2_out, hop_length=hop, win_length=win)

    x_out = np.stack((x1_out, x2_out), axis=1)
    
    return x_out

ss = spectral_subtraction

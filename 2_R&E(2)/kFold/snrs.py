import numpy as np
import soundfile as sf
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for this in ['1', '2', '3', '4', '5', '6']:
    wav_list_original = np.sort(glob('/home/rne/Documents/baseline/Data/Original_Data/*.wav'))
    wav_list_noise = np.sort(glob('/home/rne/Documents/baseline/Data/Audio_with_Noise0/*_'+this+'.wav'))
    wav_list = list(zip(wav_list_original, wav_list_noise))

    SNRs = []

    for (original, noise) in wav_list:
        original_wav, _ = sf.read(original)
        noise_wav, _ = sf.read(noise)
        difference = original_wav - noise_wav
        signal_Arms = np.sum(original_wav**2)
        noise_Arms = np.sum(difference**2)
        SNR = 10 * np.log10(signal_Arms/noise_Arms)
        SNRs.append(SNR)

    SNRs = np.array(SNRs, dtype=np.float32)
    print(this)
    print('mean', np.mean(SNRs))
    print('median', np.median(SNRs))
    print('min', np.min(SNRs))
    print('max', np.max(SNRs))
    print('std', np.std(SNRs))
    print()
    print('______________________________')
    SNRs = np.array([SNRs], dtype=float)
    dataset = pd.DataFrame({'SNR': SNRs[0, :]})
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.axis([-25, 5, None, None])
    sns.violinplot(x=dataset['SNR'])
    plt.tight_layout()
    plt.savefig('./fuck'+this+'.png',  dpi=600)
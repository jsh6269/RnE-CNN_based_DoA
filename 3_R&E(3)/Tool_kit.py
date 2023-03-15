import numpy as np
from glob import glob
import os
import inspect
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from os.path import join
from tqdm import tqdm
import torchaudio
import librosa.display
import torch

def info_writing(mainFile, modelClass, savepath):
    """
    ex) info_writing(__file__, DoANet)
    :param mainFile: source file of main method, please
    :param modelClass: module of model (class structure)
    :param savepath: savepath of info text files
    :return: nothing. It will
    """
    with open(mainFile, 'r') as f:
        A = f.readlines()

    with open(savepath + '/Model_Info.txt', 'w') as f:
        for val in A:
            if 'import' in val or '# parser' in val:
                pass
            elif '# end' in val:
                break
            else:
                if '#' not in val:
                    f.write(val)
        f.write('\n')

    with open(inspect.getfile(modelClass), 'r') as f:
        A = f.readlines()

    with open(savepath + '/Model_Info.txt', 'a') as f:
        for val in A:
            f.write(val)


def print_out_doa(which_set, epoch, corr, tot, loss, timeDelayed, savepath, dict={}):
    """
    :param which_set, epoch, corr, tot, loss, timeDelayed
    :param savepath: savepath of result text files
    :return: nothing. it prints out the current result and it also writes result text files
    """
    # = {
    #   3 : {'gt' : 2, 'pred' : [?, ?, ?, ?]}
    # }
    #
    # if id=3, gt=2, pred=[?, ?, ?, ?]

    acc = float(corr) / tot

    if dict:
        total_correct = 0
        total = len(list(dict.keys()))
        for item in dict.values():
            if item['gt'] == max(item['pred'], key=item['pred'].count):
                total_correct += 1
        total_acc = total_correct / total
        stdout = '{} {}: [{:6.0f}/{:6.0f} ({:2.0f}%)]   Loss: {:.3f}   Acc: {:.3f}   tot_Acc: {:.3f} ({:2.0f}%)   Time: {:.3f} sec'.format(
            which_set.capitalize(), epoch, corr, tot, 100. * acc, loss, acc, total_acc, total_acc * 100, timeDelayed)

        with open(join(savepath, 'tot_acc.txt'), 'a') as f:
            f.write('{}\t{:.5f}\n'.format(epoch, total_acc))

    else:
        stdout = '{} {}: [{:6.0f}/{:6.0f} ({:2.0f}%)]   Loss: {:.3f}   Acc: {:.3f}   Time: {:.3f} sec'.format(
            which_set.capitalize(), epoch, corr, tot, 100. * acc, loss, acc, timeDelayed)

    print(stdout)

    with open(join(savepath, 'total.txt'), 'a') as f:
        f.write('{}\n'.format(stdout))

    if which_set != "test" and 'iter' not in which_set:
        with open(join(savepath, 'loss_' + which_set + '.txt'), 'a') as f:
            f.write('{}\t{:.5f}\n'.format(epoch, loss))

        with open(join(savepath, 'acc_' + which_set + '.txt'), 'a') as f:
            f.write('{}\t{:.5f}\n'.format(epoch, acc))


def print_out_vad(which_set, epoch, corr, tot, loss, precesion, recall, F1_score, timeDelayed, savepath):
    acc = float(corr) / tot
    stdout = '{} {}: [{:5.0f}/{:5.0f} ({:2.0f}%)]  F1_score: {:.3f}\tLoss: {:.3f}   Precesion: {:.2f}   ' \
             'Recall: {:.2f}   Acc: {:.3f}  Time: {:.3f} sec'.format(which_set.capitalize(), epoch, corr, tot,
              100. * acc, F1_score, loss, precesion, recall, acc, timeDelayed)
    print(stdout)

    with open(join(savepath, 'total.txt'), 'a') as f:
        f.write('{}\n'.format(stdout))

    if which_set != "test":
        with open(join(savepath, 'loss_' + which_set + '.txt'), 'a') as f:
            f.write('{}\t{:.5f}\n'.format(epoch, loss))

        with open(join(savepath, 'acc_' + which_set + '.txt'), 'a') as f:
            f.write('{}\t{:.5f}\n'.format(epoch, acc))

        with open(join(savepath, 'f1score_' + which_set + '.txt'), 'a') as f:
            f.write('{}\t{:.5f}\n'.format(epoch, F1_score))


def interpret(file, cutoff=0):
    """
    special code for "result_plot"
    """
    fp1 = open(file, "r")
    epoch, result = [], []
    while True:
        data = fp1.readline()
        if not data:
            break
        data = data.split('\t')
        epoch.append(float(data[0]))
        data[1] = data[1][:-1]
        result.append(float(data[1]))
    epoch = epoch[cutoff:]
    result = result[cutoff:]
    return epoch, result


def result_plot(datapath, cutoff=0, f1score=False):
    """
    :param cutoff: amount of the very first few epochs to cutoff
    :param datapath: datapath to peek the result and save the plot
    :return: nothing. it saves the result plot
    """
    x1, y1 = interpret(join(datapath, 'loss_train.txt'), cutoff=cutoff)
    x2, y2 = interpret(join(datapath, 'loss_valid.txt'), cutoff=cutoff)
    x3, y3 = interpret(join(datapath, 'acc_train.txt'), cutoff=cutoff)
    x4, y4 = interpret(join(datapath, 'acc_valid.txt'), cutoff=cutoff)

    plt.figure(figsize=(48, 12))
    plt.subplot(121)
    plt.plot(x1, y1, color='blue', label='loss_train')
    plt.plot(x2, y2, color='red', label='loss_valid')
    plt.legend(['loss_train', 'loss_valid'], fontsize=32)
    plt.subplot(122)
    plt.plot(x3, y3, color='green', label='acc_train')
    plt.plot(x4, y4, color='orange', label='acc_valid')
    plt.legend(['acc_train', 'acc_valid'], fontsize=32)
    plt.savefig(join(datapath, 'plot.png'), dpi=100)
    plt.close()

    if f1score:
        x5, y5 = interpret(join(datapath, 'f1score_train.txt'), cutoff=cutoff)
        x6, y6 = interpret(join(datapath, 'f1score_valid.txt'), cutoff=cutoff)
        plt.figure(figsize=(24, 12))
        plt.plot(x5, y5, color='blue', label='F1_score_train')
        plt.plot(x6, y6, color='red', label='F1_score_valid')
        plt.legend(['F1_score_train', 'F1_score_valid'], fontsize=32)
        plt.savefig(join(datapath, 'plot2.png'), dpi=100)
        plt.close()


def compare_plot(datapaths, savepath, cutoff=0, f1score=False):
    """
    :param datapaths: datapaths
    :param savepath: savepath
    :param cutoff: cutoff
    :param f1score: use f1score or not (boolean)
    :return:
    """
    loss_train, loss_valid, acc_train, acc_valid, f1_train, f1_valid = [],[], [], [], [], []
    namelist = ['loss_train', 'loss_valid', 'acc_train', 'acc_valid', 'f1_train', 'f1_valid']

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for datapath in datapaths:
        loss_train.append(interpret(join(datapath, 'loss_train.txt'), cutoff=cutoff))
        loss_valid.append(interpret(join(datapath, 'loss_valid.txt'), cutoff=cutoff))
        acc_train.append(interpret(join(datapath, 'acc_train.txt'), cutoff=cutoff))
        acc_valid.append(interpret(join(datapath, 'acc_valid.txt'), cutoff=cutoff))

        if f1score:
            f1_train.append(interpret(join(datapath, 'f1score_train.txt'), cutoff=cutoff))
            f1_valid.append(interpret(join(datapath, 'f1score_valid.txt'), cutoff=cutoff))

    L = [loss_train, loss_valid, acc_train, acc_valid, f1_train, f1_valid]

    for index in range(len(L)):
        plt.figure(figsize=(24, 12))
        plt.title(namelist[index])
        for file in range(len(L[index])):
            plt.plot(L[index][file][0], L[index][file][1], label=os.path.basename(datapaths[file]))
        plt.legend([os.path.basename(datapaths[file]) for file in range(len(L[index]))])
        plt.savefig(join(savepath, namelist[index]))
        plt.close()


def get_next_epoch(savepath):

    if not os.path.exists(join(savepath, 'total.txt')):
        return 0

    with open(join(savepath, 'total.txt'), 'r') as f:
        last = f.readlines()[-1]

        if 'iter' in last:
            print("iteration status is not acceptable")
            raise AssertionError

        which_set, epoch = last[:5], int(last.split(':')[0][5:])
        which_set = which_set.lower()
        if which_set == 'train':
            return epoch + 1

        # acc_valid.txt, loss_valid.txt, total.txt
        if which_set == 'valid':
            remove_last(join(savepath, 'acc_valid.txt'))
            remove_last(join(savepath, 'loss_valid.txt'))
            remove_last(join(savepath, 'total.txt'))
            return epoch


def remove_last(path):
    """
    remove the last line of text file
    """
    with open(path, 'r') as f:
        A = f.readlines()
    with open(path, 'w') as f:
        for line in A[:-1]:
            f.write(line)


def check_num(path):
    """
    :param path: path of vadGt.npy
    :return: nothing it prints Total yes, no, ratio and minYes
    """
    x = np.load(path)
    names = get_name()
    Yes = 0
    No = 0
    minyes = 9999
    minName = []
    for i in range(len(x)):
        yes, no = 0, 0
        for val in x[i]:
            if val == 0:
                no = no + 1
            else:
                yes = yes + 1
        if minyes > yes:
            minyes = yes
        Yes += yes
        No += no

    for i in range(len(x)):
        yes = 0
        for val in x[i]:
            if val != 0:
                yes = yes + 1
        if yes == 4:
            minName.append(names[i])

    print("Total yes:", Yes, "\nTotal no:", No)
    print("{:.8f}\t{:.8f}".format(Yes / (Yes + No), No / (Yes + No)))
    print("minYes:", minyes)
    print("minName:", minName)


def logmel(x, sr, quality='low'):
    global hop
    if quality == 'high':
        mel_dim = 210
    else:
        mel_dim = 80

    hop = int(8 / 1000 * sr)
    win = int(32 / 1000 * sr)

    X0 = librosa.feature.melspectrogram(y=x[:, 0], sr=sr, n_mels=mel_dim, hop_length=hop, n_fft=win)
    X1 = librosa.feature.melspectrogram(y=x[:, 1], sr=sr, n_mels=mel_dim, hop_length=hop, n_fft=win)
    X0 = librosa.power_to_db(X0)
    X1 = librosa.power_to_db(X1)
    return X0, X1


def spec_plot(datapath, savepath):
    """
    :param datapath: './rneData2/2XAudio/*.wav'
    :param savepath: './specplot'
    :return: nothing. it draws specplot to your savepath, make path if it doesn't exist
    """
    wavlist = np.sort(glob(datapath))
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for wav in tqdm(wavlist):
        file_name = os.path.basename(wav)[:-4]
        waveform, sr = torchaudio.load(wav)
        if len(waveform[0]) == 0:
            print(file_name)
            plt.figure(figsize=(5, 9))
            plt.plot()
            plt.savefig(savepath + file_name + '.png')
            plt.close()
            continue
        xmax = len(waveform[1])
        plt.figure(figsize=(5, 9))
        plt.subplot(411)
        plt.xlim((0, xmax))
        plt.ylim((-0.12, 0.12))
        plt.plot(waveform[0].t().numpy())
        plt.subplot(412)
        plt.xlim((0, xmax))
        plt.ylim((-0.12, 0.12))
        plt.plot(waveform[1].t().numpy(), c='#ff7f0e')

        x, sr = sf.read(wav)
        mel_L, mel_R = logmel(x, sr, 'high')
        plt.subplot(413)
        hop = int(8 / 1000 * sr)
        librosa.display.specshow(mel_L, sr=sr, hop_length=hop, y_axis='mel', x_axis='time', fmax=8000, cmap='magma')
        plt.subplot(414)
        librosa.display.specshow(mel_R, sr=sr, hop_length=hop, y_axis='mel', x_axis='time', fmax=8000, cmap='magma')

        plt.savefig(join(savepath, file_name + '.png'))
        plt.close()


def get_name(datapath='./rneData2/OriginalAudio/*.wav'):
    """
    :param datapath: default: './rneData2/Original_Audio/*.wav'
    :return: list file that contains names of files according to the datapath
    """
    datalist = np.sort(glob(datapath))
    X = []
    for file in datalist:
        X.append(os.path.basename(file))
    return X


def mel_plot(npData, savepath, sr=48000):
    """
    :param npData: './rneData2/2XAudioNoise/bees.npy'
    :param savepath: './mel_plot'
    :return: nothing it draws mel_plot according to the npData package and save it to savepath makes path if it doesn't exist
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    X = np.load(npData)
    filename = get_name()
    hop = int(8 / 1000 * sr)
    for index in tqdm(range(len(X))):
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        librosa.display.specshow(X[index][0], sr=sr, hop_length=hop, y_axis='mel', x_axis='time', fmax=8000, cmap='magma')
        plt.subplot(122)
        librosa.display.specshow(X[index][1], sr=sr, hop_length=hop, y_axis='mel', x_axis='time', fmax=8000, cmap='magma')
        plt.savefig(join(savepath, filename[index] + '.png'))
        plt.close()


def commander(x):
    # noinspection All
    X = torch.empty(size=(x.shape[0], 39, 2, 80, 6), dtype=torch.float32)
    for i in range(x.shape[0]):
        X[i] = get_fraction(x[i])
    return X


def get_fraction(x):
    melLen, splitNum = 244, 39
    # noinspection All
    output = torch.empty(size=(39, 2, 80, 6))
    for i in range(splitNum):
        start = int(i * (melLen / splitNum))
        finish = start + 6
        output[i] = x[:, :, start:finish]
    return output

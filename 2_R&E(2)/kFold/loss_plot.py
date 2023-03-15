import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def interpret(file, cutoff=0):
    fp1= open(file, "r")
    epoch = []
    loss = []
    while True:
        data = fp1.readline()
        if not data: break
        data = data.split('\t')
        epoch.append(float(data[0]))
        data[1] = data[1][:-1]
        loss.append(float(data[1]))
    epoch = epoch[cutoff:]
    loss = loss[cutoff:]
    return epoch, loss

c=0
from glob import glob
folder = np.sort(glob('./jowonjun/*'))
for i in range(len(folder)):
    path = folder[i] + '/'

    x1, y1 = interpret(path + 'loss_train.txt', cutoff=c)
    x2, y2 = interpret(path + 'loss_valid.txt', cutoff=c)
    x3, y3 = interpret(path + 'acc_train.txt', cutoff=c)
    x4, y4 = interpret(path + 'acc_valid.txt', cutoff=c)

    plt.figure(figsize=(48, 12))
    plt.subplot(121)
    plt.plot(x1, y1, color='blue', label = 'loss_train')
    plt.plot(x2, y2, color='red', label = 'loss_valid')
    plt.legend(['loss_train', 'loss_valid'], fontsize = 32)
    plt.subplot(122)
    plt.plot(x3, y3, color='green', label = 'acc_train')
    plt.plot(x4, y4, color='orange', label ='acc_valid')
    plt.legend(['acc_train', 'acc_valid'], fontsize = 32)
    plt.savefig(path + 'plot.png', dpi=100)
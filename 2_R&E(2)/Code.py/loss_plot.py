import numpy as np
from matplotlib import pyplot as plt

def interpret(file, cutoff=2):
    fp1= open(file, "r")
    loss = []
    while True:
        data = fp1.readline()
        if not data: break
        data = data.split('\t')
        data[1] = data[1][:-1]
        loss.append(float(data[1]))
    loss = loss[cutoff:]
    return loss

c=10
y1 = interpret('./loss_train.txt', cutoff=c)
x1 = np.arange(1+c,len(y1)+1+c,1)
y2 = interpret('./loss_valid.txt', cutoff=c)
x2 = np.arange(1+c,len(y2)+1+c,1)

plt.plot(x1, y1, color='blue')
plt.plot(x2, y2, color='red')
plt.show()
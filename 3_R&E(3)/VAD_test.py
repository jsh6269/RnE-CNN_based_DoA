import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from VAD_model import VADNet
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import argparse
from Tool_kit import *

parser = argparse.ArgumentParser(description='VAD test')
parser.add_argument('--testMel', type=str, default='./rneData2/16XAudioNoise/bees.npy',)
parser.add_argument('--vadGt', type=str, default='./rneData2/vadGt.npy')
parser.add_argument('--ptPath', type=str, default='./previous_pt/max_f1score_model.pt')
parser.add_argument('--savepath', type=str, default='./testResult')
args = parser.parse_args()

npData = np.load(args.testMel)
gtData = np.load(args.vadGt)
filelist = get_name()

if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

model = VADNet()
model.cuda()
model.load_state_dict(torch.load(args.ptPath))
model.eval()
predList = []

# noinspection All
classWeight = torch.tensor([0.318435, 0.681565]).to("cuda")
celoss = nn.CrossEntropyLoss(weight=classWeight)
loss = 0
TP, FP, FN, TN = 0, 0, 0, 0
# noinspection All
gtData = torch.tensor(gtData, dtype=torch.int64).cuda()
# noinspection All
npData = commander(torch.tensor(npData, dtype=torch.float32)).cuda()

with torch.no_grad():
    # shape of vadGt : (diverse, 1160, 39)
    X = torch.empty(size=(args.diverse, 1160, 39), dtype=torch.int64)
    for i in range(args.diverse):
        for j in range(1160):
             "Write something here"
    # We should make vadGt from this sourcefile
    # We prolong the procedure since there's not much time


    for i in tqdm(range(1160)):
        data = npData[i]
        filename = filelist[i]
        pred = model(data)
        loss += celoss(pred, gtData[i])
        pred = torch.argmax(pred, dim=1).to('cpu')
        # compare pred, gt[i]
        err = 0
        for j in range(len(pred)):
            p, g = pred[j], gtData[i][j]
            if p == 1 and g == 1:
                TP += 1
            elif p == 1 and g == 0:
                FP += 1
                err += 1
            elif p == 0 and g == 1:
                FN += 1
                err += 1
            elif p == 0 and g == 0:
                TN += 1
        predList.append([filename, list(np.asarray(pred)), err, list(np.asarray(gtData[i].to('cpu')))])

    predList.sort()
    corr = TP + TN
    tot = TP + FP + FN + TN
    acc = corr / tot
    precesion = 0 if (TP + FP == 0) else TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 0 if (precesion + recall == 0) else 2 * precesion * recall / (precesion + recall)

    stdout = 'Test result: [{:5.0f}/{:5.0f} ({:2.0f}%)]  F1_score: {:.3f}\tLoss: {:.3f}   Precesion: {:.2f}   ' \
             'Recall: {:.2f}   Acc: {:.3f}'.format(corr, tot,
              100. * acc, F1_score, loss / 1160, precesion, recall, acc)

    with open(join(args.savepath, 'VAD_result.txt'), 'w') as f:
        f.write('{}\n\n'.format(stdout))
        for i in range(1160):
            f.write('{} : {} wrongs\n'.format(predList[i][0], predList[i][2]))
            f.write("pred : {}\ngt   : {}\n\n".format(predList[i][1], predList[i][3]))

    predList.sort(key=lambda x : x[2], reverse=True)
    with open(join(args.savepath, 'VAD_error.txt'), 'w') as f:
        f.write('{}\n\n'.format(stdout))
        for i in range(1160):
            f.write('{} : {} wrongs\n'.format(predList[i][0], predList[i][2]))
            f.write("pred : {}\ngt   : {}\n\n".format(predList[i][1], predList[i][3]))


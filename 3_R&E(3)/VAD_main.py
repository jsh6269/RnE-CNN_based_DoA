import time, os, random, math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from VAD_model import VADNet
from VAD_dataset import Dataset
from os.path import join
from Tool_kit import *

parser = argparse.ArgumentParser(description='Voice Activity Detection Network')
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--savepath', type=str, default='default')
parser.add_argument('--kindex', type=int, default=1)
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--ptPath', type=str, default='./model_saved/min_loss_model.pt')
parser.add_argument('--npPath', type=list, default=[
    './rneData2/8XAudioNoise/drone1'
])
parser.add_argument('--gtPath', type=str, default='./rneData2/vadGt.npy')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--pin_memory', type=bool, default=True)
parser.add_argument('--validate', type=list, default=[
    './rneData2/8XAudioNoise/drone1'
])

args = parser.parse_args()

# for folder in np.sort(glob('./rneData2/*XAudioNoise/*')):
#     if '.npy' in folder:
#         continue
#     if 'street' not in folder and 'Siren' not in folder:
#         args.npPath.append(folder)
# end

args.cuda = torch.cuda.is_available()
kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

min_loss = 1e8
max_f1score = 0
max_acc = 0
start = time.time()
cnt = 0
# noinspection All

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.savepath == 'default':
    today = time.strftime('%y%m%d (%Hh %Mm %Ss)')
    args.savepath = join('result', '{}{}'.format(today, './'))

if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)


def main_loop(epoch, which_set):

    global min_loss
    global max_f1score
    global max_acc
    global start
    global classWeight
    global vadPred, cnt

    if which_set == 'train':
        model.train()
    else:
        model.eval()

    if args.cuda:
        classWeight = torch.tensor([0.33222812, 0.66777188]).to("cuda")
    else:
        classWeight = torch.tensor([0.33222812, 0.66777188])

    celoss = nn.CrossEntropyLoss(weight=classWeight)
    epoch_loss = 0
    TP, FP, FN, TN = 0, 0, 0, 0

    for batch_idx, (data, gt, IDs) in enumerate(loader[which_set]):
        IDs = torch.stack(IDs, dim=-1)
        # data ; [batch, 2, 80, 6]
        # gt ; [batch]
        data, gt = data.cuda(), gt.cuda()

        optimizer.zero_grad()
        pred = model(data)
        loss = celoss(pred, gt)

        if which_set == 'train':
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            torch.save(model.state_dict(), join(args.savepath, 'model.pt'))
            epoch_loss += loss.data
            pred = torch.argmax(pred, dim=1)

            for i in range(len(pred)):
                if (pred[i], gt[i]) == (1, 1):
                    TP += 1
                elif (pred[i], gt[i]) == (1, 0):
                    FP += 1
                elif (pred[i], gt[i]) == (0, 1):
                    FN += 1
                elif (pred[i], gt[i]) == (0, 0):
                    TN += 1
                if which_set == 'test':
                    # diverse, dataNum, start, finish, gt, intervalNum
                    tmp = IDs[i]
                    vadPred[tmp[0]][tmp[1]][tmp[5]] = pred[i].type(torch.int64)
                    # cnt += 1
                    # print(cnt/((len(args.npPath)+len(args.validate)) * 1180 * 39))
                    # vadPred[diverse][dataNum][interval] = pred[i]

    with torch.no_grad():
        epoch_correct = TP + TN
        precesion = 0 if (TP + FP == 0) else TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = 0 if (precesion + recall == 0) else 2 * precesion * recall / (precesion + recall)

        epoch_loss = epoch_loss / (len(loader[which_set]))
        epoch_acc = epoch_correct / len(loader[which_set].dataset)
        print_out_vad(which_set, epoch, epoch_correct, len(loader[which_set].dataset), epoch_loss, precesion, recall, F1_score, time.time()-start, args.savepath)
        start = time.time()

        if which_set == 'valid' and min_loss > epoch_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), join(args.savepath, 'min_loss_model.pt'))

        if which_set == 'valid' and max_f1score < F1_score:
            max_f1score = F1_score
            torch.save(model.state_dict(), join(args.savepath, 'max_f1score_model.pt'))

        if which_set == 'valid' and max_acc < epoch_acc:
            max_acc = epoch_acc
            torch.save(model.state_dict(), join(args.savepath, 'max_acc_model.pt'))

if __name__=='__main__':

    loader = {
        'train': torch.utils.data.DataLoader(Dataset(which_set='train', kfold=args.kfold, kindex=args.kindex,
        npPath=args.npPath, gtPath=args.gtPath, gpu=args.cuda), batch_size=args.batch_size, shuffle=True, **kwargs),
        'valid': torch.utils.data.DataLoader(Dataset(which_set='valid', kfold=args.kfold, kindex=args.kindex,
        npPath=args.validate, gtPath=args.gtPath, gpu=args.cuda), batch_size=args.batch_size, shuffle=True, **kwargs),
        'test': torch.utils.data.DataLoader(Dataset(which_set='test', kfold=args.kfold, kindex=args.kindex,
        npPath=args.validate+args.npPath, gtPath=args.gtPath, gpu=args.cuda), batch_size=args.batch_size, shuffle=True, **kwargs)
        # 'test' : concat valid and train
    }

    model = VADNet()
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    info_writing(__file__, VADNet, args.savepath)

    if args.mode == 'train':
        for epoch in range(args.epochs):
            with torch.no_grad():
                main_loop(epoch, which_set='valid')
            main_loop(epoch, which_set='train')
            if epoch != 0 and epoch % 5 == 0:
                result_plot(args.savepath, f1score=True)

    if args.mode == 'test':
        if args.cuda:
            model.load_state_dict(torch.load(args.ptPath))
        else:
            model.load_state_dict(torch.load(args.ptPath, map_location='cpu'))
        vadPred = torch.zeros(size=(len(args.npPath)+len(args.validate), 1180, 39), dtype=torch.int64).cuda()
        main_loop(1, which_set='test')
        torch.save(vadPred, './vadPred/vadPred2.pt')

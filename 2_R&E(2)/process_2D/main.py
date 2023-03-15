import time, os, random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import argparse, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch, math
from torch.backends import cudnn
from model import DoANet
from dataset import Dataset
from os.path import join
from glob import glob

parser = argparse.ArgumentParser(description='Speaker Embedding Network')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--kindex', type=int, default=1)
parser.add_argument('--kfold', type=int, default=3)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--partition', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--input-channel', type=int, default=8)
parser.add_argument('--ptname', type=str, default='bestmodel.pt')
parser.add_argument('--datapath', type=str, default='./wavft_test')
# parser.add_argument('--gpu_num', type=str, default='1')

# lr,

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

filelist = glob(args.datapath + '/*.npy')
filenum = len(filelist)
args.batch_size = int(math.ceil(filenum*(args.kfold-1)/(args.kfold*args.partition)))

min_train_loss = 1e8
min_valid_loss = 1e8

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

today = time.strftime('%y%m%d (%Hh %Mm %Ss)')
savepath = join('result', '{}{}'.format(today, './'))

if not os.path.exists(savepath):
    os.makedirs(savepath)
else: raise FileExistsError

def clock():
    today = time.strftime('%Hh %Mm %Ss')
    print(today)

def main_loop(epoch, which_set):

    global min_train_loss
    global min_valid_loss

    if which_set == 'train':
        model.train()
    else:
        model.eval()

    celoss = nn.CrossEntropyLoss()
    epoch_loss = 0
    epoch_correct = 0

    for batch_idx, (data, gt, fpath) in enumerate(loader[which_set]):
        if args.cuda:  data, gt = data.cuda(), gt.cuda()
        optimizer.zero_grad()
        pred = model(data)
        loss = celoss(pred, gt)

        if which_set == 'train':
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), join(savepath, 'model.pt'))

        if which_set == 'train':
            # saving train result in txt file
            num_tbatch = len(loader['train'])

            with open(join(savepath, 'loss_train.txt'), 'a') as f:
                f.write('{:.2f}\t{:.5f}\n'.format(epoch + float(batch_idx) / num_tbatch, loss.data))

            pred = torch.argmax(pred, dim=1)
            corr = (pred == gt).sum().float()
            acc = corr / len(pred)

            with open(join(savepath, 'acc_train.txt'), 'a') as f:
                f.write('{:.2f}\t{:.5f}\n'.format(epoch + float(batch_idx) / num_tbatch, acc))

            stdout = 'Train Epoch: {} [{:5.0f}/{:5.0f} ({:2.0f}%)]\tLoss: {:.3f} Acc: {:.3f}'.format(
                    epoch + 1, corr, len(pred), 100. * acc, loss.data, acc)
            print(stdout)

            with open(join(savepath, 'total.txt'), 'a') as f:
                f.write('train  {}  {}\n'.format(epoch + 1, stdout))
                # f.write(str(pred))
                # f.write(str(gt))

            if min_train_loss > loss.data:
                torch.save(model.state_dict(), join(savepath, 'least_train_loss_model.pt'))
                min_train_loss = loss.data

        if which_set == 'valid' or which_set == 'test':
            epoch_loss += loss.data
            pred = torch.argmax(pred, dim=1)
            epoch_correct += (pred == gt).sum()

    # saving test result in txt file
    if which_set == 'test':
        with open(join('./previous_pt/'+args.ptname[:-2]+'_test_result.txt'), 'a') as f:
            epoch_loss = epoch_loss / len(loader['test'])
            epoch_acc = float(epoch_correct) / len(loader['test'].dataset)
            f.write('{:.5f}\t{:.5f}  {:.5f}%\n'.format(epoch_loss, epoch_acc, epoch_acc*100))
            f.write(str(pred))
            f.write('\n')
            f.write(str(gt))
            f.write('\n')

    if which_set == 'valid':
        # saving valid result in txt file
        with open(join(savepath, 'loss_valid.txt'), 'a') as f:
            epoch_loss = epoch_loss / len(loader['valid'])
            f.write('{}\t{:.5f}\n'.format(epoch, epoch_loss))

        with open(join(savepath, 'acc_valid.txt'), 'a') as f:
            epoch_acc = float(epoch_correct) / len(loader['valid'].dataset)
            f.write('{}\t{:.5f}\n'.format(epoch, epoch_acc))

        stdout = 'Valid Epoch: {} [{:5.0f}/{:5.0f} ({:2.0f}%)]\tLoss: {:5.3f} Acc: {:5.3f}'.format(
            epoch, epoch_correct, len(loader[which_set].dataset), 100*epoch_acc, epoch_loss, epoch_acc)
        print(stdout)

        with open(join(savepath, 'total.txt'), 'a') as f:
            f.write('{}  {}  {}\n'.format(which_set, epoch, stdout))
            # f.write(str(pred))
            # f.write(str(gt))

        if min_valid_loss > epoch_loss:
            min_valid_loss= epoch_loss
            torch.save(model.state_dict(), join(savepath, 'least_valid_loss_model.pt'))

if __name__=='__main__':

    loader = {
        'train': torch.utils.data.DataLoader(Dataset(which_set='train', kfold=args.kfold, kindex=args.kindex,
        datapath= args.datapath, order = 1), batch_size=args.batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(Dataset(which_set='valid', kfold=args.kfold, kindex=args.kindex,
        datapath = args.datapath, order = 2), batch_size=args.batch_size, shuffle=True),
    }
    import os

    model = DoANet(args.input_channel)
    if args.cuda: model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    count=0
    for name, param in model.named_parameters():
        if 'fc1' in name or 'cbl2' in name:
            paramrequires_grad = False
            print(name)

    # for params in model.parameters():
    #     params.requires_grad = False

    if args.mode == 'train':
        for epoch in range(args.epochs):
            main_loop(epoch, which_set='valid')
            main_loop(epoch, which_set='train')

    if args.mode == 'test':
        if args.cuda: model.load_state_dict(torch.load('./previous_pt/'+args.ptname))
        else: model.load_state_dict(torch.load('./previous_pt/'+args.ptname, map_location='cpu'))
        main_loop(1, which_set='test')

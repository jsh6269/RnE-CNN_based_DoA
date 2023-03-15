import time, os, random, math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from DoA_ori_model import DoANet
from DoA_ori_dataset import Dataset
from Tool_kit import *
from os.path import join
from glob import glob

parser = argparse.ArgumentParser(description='Direction Of Arrival Network_original')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--savepath', type=str, default='default')
parser.add_argument('--kindex', type=int, default=0)
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=186)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--testPt', type=str, default='./previous_pt/bestmodel.pt')
parser.add_argument('--vadPt', type=str, default='./previous_pt/model.pt')
parser.add_argument('--npPath', type=list, default=[
    './rneData2/OriginalAudio.npy'])
parser.add_argument('--fileNum', type=int, default=1160)
# parser.add_argument('--num_workers', type=int, default=0)
# parser.add_argument('--pin_memory', type=bool, default=False)
# end

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
# kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

min_loss = 1e8
start = time.time()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.mode == 'continue' and args.savepath == 'default':
    args.savepath = np.sort(glob('./result/*'))[-1]

elif args.mode == 'train':
    today = time.strftime('%y%m%d (%Hh %Mm %Ss)')
    args.savepath = join('result', '{}{}'.format(today, './'))
    os.makedirs(args.savepath)


def main_loop(epoch, which_set):

    global min_loss
    global start

    if which_set == 'train':
        model.train()
    else:
        model.eval()

    celoss = nn.CrossEntropyLoss()
    epoch_loss = 0
    epoch_correct = 0
    count = 0

    for batch_idx, (data, gt, id) in enumerate(loader[which_set]):

        optimizer.zero_grad()
        pred = model(data)
        count += 1
        loss = celoss(pred, gt)
        epoch_loss += loss.data
        iter_correct = 0
        iter_loss = loss.data

        if which_set == 'train':
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), join(args.savepath, 'model.pt'))
        pred = torch.argmax(pred, dim=1)

        for i in range(len(pred)):
            if pred[i] == gt[i]:
                iter_correct += 1
                epoch_correct += 1

        # if which_set == 'train':
        #     print_out('train_iteration', epoch, iter_correct, data.shape[0], iter_loss, time.time()-start)
        #     start = time.time()

    with torch.no_grad():
        epoch_loss = epoch_loss / count
        print_out(which_set, epoch, epoch_correct, len(loader[which_set].dataset), epoch_loss, time.time()-start, args.savepath)
        start = time.time()

        if which_set == 'valid' and min_loss > epoch_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), join(args.savepath, 'min_loss_model.pt'))


if __name__=='__main__':

    loader = {
        'train': torch.utils.data.DataLoader(Dataset(which_set='train', kfold=args.kfold, kindex=args.kindex,
        npPath=args.npPath, gpu=args.cuda, vadPt=args.vadPt), batch_size=args.batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(Dataset(which_set='valid', kfold=args.kfold, kindex=args.kindex,
        npPath=args.npPath, gpu=args.cuda, vadPt=args.vadPt), batch_size=args.batch_size, shuffle=True)
    }

    model = DoANet()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    info_writing(__file__, DoANet, args.savepath)

    if args.mode == 'continue':
        model.load_state_dict(torch.load(join(args.savepath, 'model.pt')))

    if args.mode == 'train' or args.mode == 'continue':
        start_epoch = get_next_epoch(args.savepath)
        for epoch in range(start_epoch, args.epochs):
            with torch.no_grad():
                main_loop(epoch, which_set='valid')
            main_loop(epoch, which_set='train')
            if epoch != 0 and epoch % 5 == 0:
                result_plot(args.savepath)

    if args.mode == 'test':
        if args.cuda:
            model.load_state_dict(torch.load(args.testPt))
        else:
            model.load_state_dict(torch.load(args.testPt, map_location='cpu'))
        main_loop(1, which_set='test')

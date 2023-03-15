import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse, time
import main, torch
from glob import glob
from os.path import join

parser = argparse.ArgumentParser(description='Speaker Embedding Network')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--batch-size', type=int, default = 310)
# parser.add_argument('--partition', type=int, default=6)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--input-channel', type=int, default=2)
parser.add_argument('--ptname', type=str, default='bestmodel.pt')
parser.add_argument('--datapath', type=str, default='./wavft/OriginalAudio2')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
filelist = glob(args.datapath + '/*.npy')
filenum = len(filelist)
# args.batch_size = int(math.ceil(filenum*(args.kfold-1)/(args.kfold*args.partition)))
today = time.strftime('%y%m%d (%Hh %Mm %Ss)')
savepath = join('result', '{}{}'.format(today, './'))
os.makedirs(savepath)

with open('kfold.py', 'r') as f:
    parm_info = f.readlines()
with open(savepath + '/setting.txt', 'a') as f:
    for line in parm_info:
        if line.startswith('parser'):
            f.write(line)
with open('model.py', 'r') as f:
    model_info = f.read()
with open(savepath + '/setting.txt', 'a') as f:
    f.write(model_info)

for index in range(args.kfold):
    main.run_kfold(index, args, savepath)

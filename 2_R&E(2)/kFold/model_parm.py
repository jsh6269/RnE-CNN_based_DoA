import torch
from torchsummary import summary
from model2 import DoANet

this = DoANet().cuda()
summary(this, (160, 244))

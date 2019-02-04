import torch
import torchivision
import numpy as np
import deep_net
from utils import DataSplit, Optimizer

'''
Pseudo:
    Preprocess
        Normalize
    Instantiate model
    Loop:
        Train model
        Validate model
    Test model
    
'''

if __name__ == "__main__":
    hyps = HyperCollector().hyps
    data = torchvision.datasets.CIFAR10("../../datasets/", train=True, download=True)
    data = DataSplit(data)
    data.normalize()

    net = deep_net.Net(data.train_X.shape)
    optim = Optimizer(net.state_dict(), hyps['optim_type'], hyps['lr'])

    if hyps['resume']:
        net.load_state_dict(torch.load(hyps['exp_name']+".pt"))


    for epoch in hyps['n_epochs']:
        

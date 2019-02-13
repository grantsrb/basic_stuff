import torch
import torchvision
from model import ConvNet
from hyperparams import HyperParams
from utils import DataLoader, Trainer

if __name__ == "__main__":
    hyps = HyperParams().hyps
    data = DataLoader(hyps['data_set'])
    data.normalize()

    nnet = ConvNet(data.shape, hyps)
    trainer = Trainer()

    trainer.train(nnet, data, hyps)

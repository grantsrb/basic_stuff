import torch
import torchvision
import time

class DataLoader:
    def __init__(self, dataset):
        """
        dataset - string denoting the desired torchvision dataset to be used
                    or dataset collected from lane and niru's dataloader
        """
        d = dir(dataset)
        # Using niru and lane's dataloaded datset
        if "X" in d and "y" in d:
            data = dataset
            splt_idx = int(.8*data.train_data.shape[0])
            perm = torch.randperm(data.train_data.shape[0]).long()
            self.train_X = data.X[perm[:splt_idx]]
            self.train_Y = data.y[perm[:splt_idx]]
            self.valid_X = data.X[perm[splt_idx:]]
            self.valid_Y = data.y[perm[splt_idx:]]
        # Use torchvision dataset
        else:
            if dataset is "cifar10":
                data = torchvision.datasets.CIFAR10("~/ml/datasets", train=True, download=True)
            else:
                data = torchvision.datasets.MNIST("~/ml/datasets", train=True, download=True)

            splt_idx = int(.8*data.train_data.shape[0])
            perm = torch.randperm(data.train_data.shape[0]).long()
            self.train_X = data.train_data[perm[:splt_idx]]
            self.train_Y = data.train_labels[perm[:splt_idx]]
            self.valid_X = data.train_data[perm[splt_idx:]]
            self.valid_Y = data.train_labels[perm[splt_idx:]]

        self.normalized = False
        self.mean = None
        self.std = None

    def normalize(self, data=None):
        """
        Channels of dataset should be last (..., N, H, W, C)
        """
        if self.mean is None and not self.normalized:
            self.mean = torch.FloatTensor([self.train_X[...,i].mean() for i in range(self.train_X.shape[-1])])
            self.std = torch.FloatTensor([self.train_X[...,i].std() for i in range(self.train_X.shape[-1])])
        if data is None and not self.normalized:
            self.train_X = (self.train_X - self.mean)/(self.std+1e-8)
            self.valid_X = (self.valid_X - self.mean)/(self.std+1e-8)
            self.normalized = True
        else:
            data = (data - self.mean)/(self.std+1e-8)
        return data
    
    def denormalize(self, data=None):
        """
        Channels of dataset should be last (..., N, H, W, C)
        """
        if self.mean is None and not self.normalized:
            self.mean = torch.FloatTensor([self.train_X[...,i].mean() for i in range(self.train_X.shape[-1])])
            self.std = torch.FloatTensor([self.train_X[...,i].std() for i in range(self.train_X.shape[-1])])
        if data is None and self.normalized:
            self.train_X = (self.std+1e-8)*self.train_X + self.mean
            self.valid_X = (self.std+1e-8)*self.valid_X + self.mean
            self.normalized = False
        else:
            data = (self.std+1e-8)*data + self.mean
        return data
        


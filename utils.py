import torch

class DataSplit:
    def __init__(self, data, train_p=.8):
        perm = torch.randperm(data.train_data.shape[0])
        split_idx = int(len(perm)*train_p)
        self.train_X = data.train_data[:split_idx]
        self.train_Y = data.train_labels[:split_idx]
        self.valid_X = data.train_data[split_idx:]
        self.valid_Y = data.train_labels[split_idx:]
        self.normalized = False
        self.data_mean = None
        self.data_std = None

    def normalize(self):
        if self.data_mean is None:
            self.data_mean = self.train_X.mean()
            self.data_std = self.train_X.std()
        if not self.normalized:
            self.train_X = (self.train_X - self.data_mean)/(self.data_std + 1e-10)
            self.valid_X = (self.valid_X - self.data_mean)/(self.data_std + 1e-10)

    def denormalize(self):
        if self.normalized:
            self.train_X = self.train_X*(self.data_std + 1e-10) + self.data_mean
            self.valid_X = self.valid_X*(self.data_std + 1e-10) + self.data_mean

class Optimizer:
    def __init__(self, net, hyps):
        self.net = net
        self.base_lr = hyps['lr']
        self.opt_type = hyps['optim_type']
        self.optim = self.new_optimizer(self.net.state_dict(), self.opt_type, self.base_lr)

    def cycle_lr(self, t):
        """
        Sets the learning rate based off of an inverted cosine wave that has a min at y = base_lr and t = 0.

        t - the location along the x dimension to evaluate the learning rate
        """
        

    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def new_optimizer(self, state_dict, opt_type, lr):
        if 'dam' in opt_type:
            return torch.optim.Adam(state_dict, lr=lr)
        if 'prop' in opt_type:
            return torch.optim.RMSprop(state_dict, lr=lr)
        else:
            return torch.optim.Adam(state_dict, lr=lr)










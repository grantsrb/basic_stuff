import torch
import torchvision
import time
from logger import Logger, SaveIO
import torch.nn.functional as F

class Trainer:
    def __init__(self):
        pass
        
    def train(self, nnet, data, hyps):
        """
        nnet - pytorch model
        data - DataLoader object 
        hyps - dict containing hyperparameters
        """
        logger = Logger(hyps)
        io = SaveLoadIO(hyps)
        optim = self.new_optim(nnet, hyps)
        if hyps['resume']:
            io.load_module(nnet, hyps['resume_net_file'])
            io.load_module(optim, hyps['resume_optim_file'])
        b_size = hyps['batch_size']
        lossfxn = self.get_loss_fxn(hyps['loss_type'])
        n_batches = data.train_X.shape[0]//b_size
        for epoch in hyps['n_epochs']:
            start_time = time.time()
            perm = torch.randperm(data.train_X.shape[0]).long()
            acc_sum = 0
            loss_sum = 0
            for i in range(n_batches):
                optim.zero_grad()
                idxs = perm[i*b_size:(i+1)*b_size]
                x,y = Variable(data.train_X[idxs]), Variable(data.train_Y[idxs])
                preds = nnet(x)
                loss = lossfxn(preds, y)
                loss.backward()
                optim.step()
                t = i/(n_batches-1)
                acc = self.get_acc(preds, y)
                acc_sum += acc.item()
                loss_sum += loss.item()
                print("Progress:", round(t,2)*100, "% - Loss:", loss.item(), " - Acc:", acc.item(), end="\r")
            if epoch % hyps['save_period'] == 0:
                io.save_imodel(nnet, optim)
            stats = {
                "epoch":epoch,
                "loss":loss_sum/n_batches, 
                "acc":acc_sum/n_batches, 
                "exec_time":time.time()-start_time
            }
            logger.print_stats(stats)
            logger.log(stats)
        return nnet

    def get_acc(self, preds, targs):
        argmaxs, _ = torch.max(preds)
        return torch.mean((argmaxs.long() == targs.long()).float())

    def get_loss_fxn(self, hyps):
        print("Currently only support cross entropy loss")
        return F.cross_entropy

    def hyper_search(self, hyps, hyp_ranges, keys, idx, trainer, search_log):
        """
        hyps - dict of hyperparameters created by a HyperParameters object
            type: dict
            keys: name of hyperparameter
            values: value of hyperparameter
        hyp_ranges - dict of ranges for hyperparameters to take over the search
            type: dict
            keys: name of hyperparameters to be searched over
            values: list of values to search over for that hyperparameter
        keys - keys of the hyperparameters to be searched over. Used to
                allow order of hyperparameter search
        idx - the index of the current key to be searched over
        trainer - trainer object that handles training of model
        """
        if idx >= len(keys):
            if 'search_id' not in hyps:
                hyps['search_id'] = 0
                hyps['exp_name'] = hyps['exp_name']+"0"
                hyps['hyp_search_count'] = np.prod([len(hyp_ranges[key]) for key in keys])
            id_ = len(str(hyps['search_id']))
            hyps['search_id'] += 1
            hyps['exp_name'] = hyps['exp_name'][:-id_]+str(hyps['search_id'])
            best_avg_rew = trainer.train(hyps)
            params = [str(key)+":"+str(hyps[key]) for key in keys]
            search_log.write(", ".join(params)+" – BestRew:"+str(best_avg_rew)+"\n")
            search_log.flush()
        else:
            key = keys[idx]
            for param in hyp_ranges[key]:
                hyps[key] = param
                hyper_search(hyps, hyp_ranges, keys, idx+1, trainer, search_log)
        return
    
    def make_hyper_range(self, low, high, range_len, method="log"):
        if method.lower() == "random":
            param_vals = np.random.random(low, high+1e-5, size=range_len)
        elif method.lower() == "uniform":
            step = (high-low)/(range_len-1)
            pos_step = (step > 0)
            range_high = high+(1e-5)*pos_step-(1e-5)*pos_step
            param_vals = np.arange(low, range_high, step=step)
        else:
            range_low = np.log(low)/np.log(10)
            range_high = np.log(high)/np.log(10)
            step = (range_high-range_low)/(range_len-1)
            arange = np.arange(range_low, range_high, step=step)
            if len(arange) < range_len:
                arange = np.append(arange, [range_high])
            param_vals = 10**arange
        param_vals = [float(param_val) for param_val in param_vals]
        return param_vals

    def new_lr(self, optim, new_lr):
        new_optim = self.new_optim(new_lr)
        new_optim.load_state_dict(optim.state_dict())
        return new_optim

    def new_optim(self, nnet, hyps):
        optim_type = hyps['optim_type']
        lr = hyps['lr']
        momentum = hyps['momentum'] # Need to implement
        if optim_type == 'rmsprop':
            new_optim = torch.optim.RMSprop(nnet.parameters(), lr=lr) 
        elif optim_type == 'adam':
            new_optim = torch.optim.Adam(nnet.parameters(), lr=lr) 
        else:
            new_optim = torch.optim.RMSprop(nnet.parameters(), lr=lr) 
        return new_optim


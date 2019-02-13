import datetime
import os

class Logger:
    def __init__(self, hyps):
        self.exp_name = hyps['exp_name']
        self.save_folder = hyps['log_folder']
        if self.save_folder[-1] == "/": self.save_folder = self.save_folder[:-1]
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
            
        now = datetime.datetime.now()
        dateandtime = "_".join([str(now.day), str(now.month), str(now.hour)+str(now.minute)])
        self.save_file = self.save_folder+"/"+hyps['exp_name'] + dateandtime + '.txt'
        self.log = open(self.save_file, 'a') 
        if hyps['resume']:
            self.log.write('\n\nResuming ' + hyps['exp_name'] + " dd_mm_hhmm:" + dateandtime + "\n")
        else:
            self.log.write('\n\n'+hyps['exp_name']+ " dd_mm_hhmm:" + dateandtime +"\n")
        self.log.flush()

    def close(self):
        self.log.close()

    def log_stats(self, info, log=None):
        if log is None:
            log = self.log
        log.write(" – ".join([key+": "+str(round(val,5)) for key,val in sorted(info.items())]+ "\n")
        log.flush()

    def open(self, file_name=None, open_type="a"):
        if file_name is None:
            file_name = self.save_file
        self.log.open(file_name, open_type)

    def print_stats(self, info):
        print(" – ".join([key+": "+str(round(val,5)) for key,val in sorted(info.items())]))


class SaveLoadIO:
    def __init__(self, hyps):
        self.exp_name = hyps['exp_name']
        self.save_folder = hyps['model_folder']
        if self.save_folder[-1] == "/": self.save_folder = self.save_folder[:-1]
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

    def save_module(self, module, save_file=None):
        """
        Saves the state dict of the argued module to file.

        module - torch nn module to save
        save_file - string name of the file to save the state_dict to
        """
        if save_file is None:
            now = datetime.datetime.now()
            save_file = self.exp_name + "_" + "_".join([str(now.day), str(now.month), str(now.hour)+str(now.minute)])
            save_file = save_file + ".pt"
        torch.save(module.state_dict(), self.save_folder + "/" + save_file)

    def save_model(self, nnet, optim, save_file_no_ext=None):
        fname = save_file_no_ext
        if fname is None:
            now = datetime.datetime.now()
            fname = self.exp_name + "_" + "_".join([str(now.day), str(now.month), str(now.hour)+str(now.minute)])
        self.save_module(nnet, save_file=fname+"_net.pt")
        self.save_module(optim, save_file=fname+"_optim.pt")

    def load_module(self, module, file_path):
        module.load_state_dict(torch.load(file_path))
        return module
    

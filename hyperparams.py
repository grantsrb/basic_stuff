import sys
import preprocessing
from models import ConvModel, FCModel, A3CModel, GRUModel
import numpy as np

class HyperParams:
    def __init__(self, arg_hyps=None):
        
        hyp_dict = dict()
        hyp_dict['string_hyps'] = {
                    "exp_name":"default",
                    "data_set":"mnist",
                    "optim_type":'rmsprop', # Options: rmsprop, adam
                    "resume_net_file": "_", # Use full path
                    "resume_optim_file": "_", # Use full path
                    "model_folder":"saved_models",
                    "log_folder":"saved_logs",
                    "loss_type":"_", # Currently only support crossentropy loss (see Trainer.get_loss_fxn)
                    }
        hyp_dict['int_hyps'] = {
                    "n_epochs": 3, # PPO update epoch count
                    "batch_size": 256, # PPO update batch size
                    'h_size':288,
                    'save_period':10, # Number of epochs per model save
                    }
        hyp_dict['float_hyps'] = {
                    "lr":0.0001,
                    "lr_low": float(1e-12),
                    "gamma":.99,
                    "max_norm":.5,
                    "momentum":.99, # Not currently implemented (see Trainer.new_optim)
                    }
        hyp_dict['bool_hyps'] = {
                    "resume":False,
                    "decay_eps": False,
                    "decay_lr": False,
                    "decay_entr": False,
                    "use_bnorm": True,
                    }
        self.hyps = self.read_command_line(hyp_dict)
        if arg_hyps is not None:
            for arg_key in arg_hyps.keys():
                self.hyps[arg_key] = arg_hyps[arg_key]

    def read_command_line(self, hyps_dict):
        """
        Reads arguments from the command line. If the parameter name is not declared in __init__
        then the command line argument is ignored.
    
        Pass command line arguments with the form parameter_name=parameter_value
    
        hyps_dict - dictionary of hyperparameter dictionaries with keys:
                    "bool_hyps" - dictionary with hyperparameters of boolean type
                    "int_hyps" - dictionary with hyperparameters of int type
                    "float_hyps" - dictionary with hyperparameters of float type
                    "string_hyps" - dictionary with hyperparameters of string type
        """
        
        bool_hyps = hyps_dict['bool_hyps']
        int_hyps = hyps_dict['int_hyps']
        float_hyps = hyps_dict['float_hyps']
        string_hyps = hyps_dict['string_hyps']
        
        if len(sys.argv) > 1:
            for arg in sys.argv:
                arg = str(arg)
                sub_args = arg.split("=")
                if sub_args[0] in bool_hyps:
                    bool_hyps[sub_args[0]] = sub_args[1] == "True"
                elif sub_args[0] in float_hyps:
                    float_hyps[sub_args[0]] = float(sub_args[1])
                elif sub_args[0] in string_hyps:
                    string_hyps[sub_args[0]] = sub_args[1]
                elif sub_args[0] in int_hyps:
                    int_hyps[sub_args[0]] = int(sub_args[1])
    
        return {**bool_hyps, **float_hyps, **int_hyps, **string_hyps}


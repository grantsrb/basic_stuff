import sys
import gc
import resource 
    
class HyperCollector:
    """
    Optional class to collect all hyperparameters into a single python dict.
    Ability to establish hyperparameter keys exists here.
    """
    
    def __init__(self):
        
        hyp_dict = dict()
        hyp_dict['string_hyps'] = {
                    "exp_name":"default", # Base name of experiment for save files
                    "optim_type":'adam', # Picks optimizer algorithm. Options: rmsprop, adam
                    }
        hyp_dict['int_hyps'] = {
                    "n_epochs": 3, # PPO update epoch count
                    "batch_size": 256, # PPO update batch size
                    }
        hyp_dict['float_hyps'] = {
                    "lr":0.0001, # Learning rate
                    "lr_low": float(1e-12), # Lower bound on lr for learning decay
                    }
        hyp_dict['bool_hyps'] = {
                    "resume":False, # If True, resumes experiment from save files
                    "decay_lr": True, # If True, performs lr decay
                    "decay_entr": True, # If True, performs entropy coeficient decay
                    }
        self.hyps = self.read_command_line(hyp_dict)

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
                    string_hyps[sub_args[0]] = str(sub_args[1])
                elif sub_args[0] in int_hyps:
                    int_hyps[sub_args[0]] = int(sub_args[1])

        return {**bool_hyps, **float_hyps, **int_hyps, **string_hyps}

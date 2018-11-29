import numpy as np
import os
import torch.optim as op
import torch.optim.lr_scheduler as lrs

from ..utilities.utilities import increment_path, increment_filename

__all__ = ["Config"]


class Config(object):
    
    
    def __init__(self, 
                # Model config
                 run_name="run", path_to_model='', model_name='model', 
                 w_init=0.01, bv_init=0.5, bh_init=None, use_cuda=False,
                 batchsize=16, beta=None, num_sample=10, overwrite=True,
                 # Training config
                 lr=0.001, wd=1e-4, momentum=0, optimizer=None,
                 lr_scheduler=None,
                 numsteps=10, numepochs=10, with_pcd=True, 
                 keepbest=True, verbose=True,
                 # Statistics config
                 graining=1, makeplot=False, path_to_history='.'
                 ):
        # Pathes
        if not overwrite: self.RUN = increment_path(os.path.join(".", run_name))
        else: self.RUN = run_name
        if not os.path.exists(self.RUN): os.makedirs(self.RUN)
        self.PATH_TO_CONFIG = os.path.join(self.RUN, "config") 
        self.PATH_TO_MODEL = os.path.join(self.RUN, path_to_model)
        self.MODEL_NAME = model_name
        self.PATH_TO_HISTORY = os.path.join(self.RUN, path_to_history)
        
        # Learning hyper parameters
        self.INIT_PARAMS = {"w": w_init, "bv": bv_init, "bh": bh_init}
        self.LEARNING_RATE = lr
        self.WEIGHT_DECAY = wd
        if optimizer is None:
            # self.OPTIMIZER = lambda p: op.Adam(p, lr=lr, 
            #                                    betas=(0.9, 0.999),
            #                                    eps=1e-08,
            #                                    weight_decay=wd)
            self.OPTIMIZER = lambda p: op.SGD(p, lr=lr, weight_decay=wd,
                                              momentum=momentum, nesterov=False)

        if lr_scheduler is None:
            self.LR_SCHEDULER = lambda o: lrs.ReduceLROnPlateau(o, mode='min',
                                                                factor=0.5,
                                                                verbose=verbose,
                                                                patience=25)
        self.USE_CUDA = use_cuda
        self.BATCHSIZE = batchsize

        # Boltzmann machine evaluation
        self.NUM_SAMPLE = num_sample
        if beta is None:
            self.BETA = np.append(np.append(
                                  np.linspace(0., 0.5, 250),
                                  np.linspace(0.5, 0.9, 2000)), 
                                  np.linspace(0.9, 1, 4000))
        else:
            self.BETA = beta


        # Boltzmann machine Training
        self.NUMSTEPS = numsteps
        self.NUMEPOCHS = numepochs
        self.KEEPBEST = keepbest
        self.WITH_PCD = with_pcd
        self.VERBOSE = verbose

        # Statistics parameters
        self.GRAINING = graining
        self.MAKEPLOT = makeplot




        return
    
    @classmethod
    def load(cls, path):
        """Load configs from path.
        
        .. seealso: 
            save_config
            
        """
        import json
        cls = cls()
        json_data=open(path).read()
        data = json.loads(json_data)
        
        for k in data:
            setattr(cls, k, data[k])
                
        return cls
    
    def save(self):
        """Save configuration in a file.
        """
        import json
        data = self.__dict__.copy()

        for k, d in self.__dict__.items():
            if type(d) != float and \
               type(d) != int and \
               type(d) != str and \
               type(d) != bool and \
               type(d) != dict:
               del data[k]

        with open(os.path.join(self.PATH_TO_CONFIG + '.cf'), 'w') as outfile:
            json.dump(data, outfile, indent=4, separators=", ")

        return

    
    def __repr__(self):
        s = "\n######## Model Configuration ########\n"
        
        data = self.__dict__
        for key in data:
            data_str = data[key].__repr__()
            s += f"{key} : {data_str} \n"
        s += "######################################\n"
        return s
    
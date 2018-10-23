import utilities as util

class Model_Config(object):
    
    
    def __init__(self, path_to_model='.', model_name='model', overwrite=False,
                       w_init=0.01, bv_init=None, bh_init=None, use_cuda=False,
                       batchsize=16, beta=None, num_sample=100,
                       ):
        # Initialization
        self.PATH_TO_MODEL = path_to_model
        self.MODEL_NAME = model_name
        if overwrite:
            self.MODEL_NAME = util.increment_filename(self.PATH_TO_MODEL,
                                                      self.MODEL_NAME)
            
        self.INIT_PARAMS = {"w": w_init, "bv": bv_init, "bh": bh_init}
        self.USE_CUDA = False
        self.BATCHSIZE = 32

        if beta is None:
            self.BETA = np.append(np.append(
                                  np.linspace(0., 0.5, 100),
                                  np.linspace(0.5, 0.9, 500)), 
                                  np.linspace(0.9, 1, 2000))
        else:
            self.BETA = beta

        self.NUM_SAMPLE = 100


        return
    
    @classmethod
    def load_config(cls, path):
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
    
    def save_config(self, path):
        """Save configuration in a file.
        """
        import json
        data = self.__dict__.copy()
            
        with open(path, 'w') as outfile:
            json.dump(data, outfile)

        return

    
    def __repr__(self):
        string += "\n######## Model Configuration ########\n"
        
        data = self.__dict__
        for key in data:
            data_str = data[key].__repr__()
            string += f"{key} : {data_str} \n"
        string += "######################################\n"
        return string
    
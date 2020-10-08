class Config:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __str__(self):
        return self.to_string()

    def to_string(self, prefix=""):
        string = ""
        for k, v in self.__dict__.items():
            if issubclass(v.__class__, Config):
                string += prefix + f"{k}:\n"
                string += "{0}\n".format(v.to_string(prefix=prefix + "\t"))
            else:
                string += prefix + f"{k}: {v.__str__()}\n"
        return string

    def get_state_dict(self):
        state_dict = {}
        for k, v in self.__dict__.items():
            if k != "_state_dict":
                if issubclass(v.__class__, Config):
                    v_dict = v.state_dict
                    for kk, vv in v_dict:
                        state_dict[k + "/" + kk] = vv
                else:
                    state_dict[k] = v
        return state_dict

    @property
    def state_dict(self):
        return self.get_state_dict()

    def merge(self, config):
        for k, v in config.__dict__.items():
            self.__dict__[k] = v

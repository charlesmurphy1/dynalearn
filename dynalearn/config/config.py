class Config:
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

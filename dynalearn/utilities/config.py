class Config:
    def __str__(self):
        return self.to_string()

    def to_string(self, prefix=""):
        string = prefix
        for k, v in self.__dict__:
            if issubclass(v.__class__, Config):
                string += f"{k}: {v.__class__}\n"

                string += "{0}\n".format(v.to_string(prefix=prefix + "\t"))
            else:
                string += f"{k}: {v}\n"
        return string

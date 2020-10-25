import tqdm
from datetime import datetime


class Verbose:
    def __init__(self, filename=None, vtype=0, notebook=False):
        self.filename = filename
        self.to_file = filename is not None
        self.vtype = vtype
        if notebook:
            self.template = tqdm.tqdm_notebook
        else:
            self.template = tqdm.tqdm
        if self.to_file:
            _file = open(self.filename, "w")
            _file.close()

    def __call__(self, msg):
        if self.to_file:
            self.save_msg(msg)
        if self.vtype != 0:
            self.print_msg(msg)

    def save_msg(self, msg):
        _file = open(self.filename, "a")
        _file.write(f"{msg}\n")
        _file.close()

    def print_msg(self, msg):
        print(msg)

    def progress_bar(self, name, num_update):
        if self.vtype == 1:
            self.save_msg(name)
            return self.template(range(num_update), name)
        else:
            self(name)
            return None

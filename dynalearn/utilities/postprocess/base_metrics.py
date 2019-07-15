import h5py


class Metrics:
    def __init__(self, verbose=1):
        self.data = dict()
        self.val_data = None
        self.test_data = None
        self.verbose = verbose
        return

    def compute(self, experiment, **kwargs):
        raise NotImplementedError()

    def display(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, h5file, overwrite=True):
        for k, v in self.data.items():
            path = self.__class__.__name__ + "/" + str(k)
            if path in h5file and overwrite:
                del h5file[path]
                h5file.create_dataset(path, data=v)
            elif path not in h5file:
                h5file.create_dataset(path, data=v)

    def load(self, h5file):
        if self.__class__.__name__ in h5file:
            self.data = self.read_h5_recursively(h5file[self.__class__.__name__])

    def read_h5_recursively(self, h5file, prefix=""):
        ans_dict = {}
        for key in h5file:
            item = h5file[key]
            if prefix == "":
                path = f"{key}"
            else:
                path = f"{prefix}/{key}"

            if isinstance(item, h5py.Dataset):
                ans_dict[path] = item[...]
            elif isinstance(item, h5py.Group):
                d = self.read_h5_recursively(item, path)
                ans_dict.update(d)
            else:
                raise ValueError()
        return ans_dict

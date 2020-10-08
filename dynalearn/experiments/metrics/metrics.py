import h5py
import tqdm

from abc import ABC, abstractmethod


class Metrics(ABC):
    def __init__(self, config, verbose=0):
        self.config = config
        self.verbose = verbose
        self.data = {}
        self.names = []
        self.get_data = {}
        self.num_updates = 0

    @abstractmethod
    def initialize(self, experiment):
        raise NotImplementedError("initialize must be implemented.")

    def exit(self, experiment):
        return

    def compute(self, experiment, verbose=None):
        self.verbose = verbose or self.verbose
        self.initialize(experiment)

        if self.verbose == 1:
            pb = tqdm.tqdm(range(self.num_updates), self.__class__.__name__)
        elif self.verbose != 0:
            print(self.__class__.__name__)
            pb = None
        else:
            pb = None

        for k in self.names:
            self.data[k] = self.get_data[k](pb)

        if self.verbose == 1:
            pb.close()

        self.exit(experiment)

    def save(self, h5file, name=None):
        if type(h5file) is not h5py.File:
            raise ValueError("Dataset file format must be HDF5.")

        name = name or self.__class__.__name__

        for k, v in self.data.items():
            path = name + "/" + str(k)
            if path in h5file:
                del h5file[path]
            h5file.create_dataset(path, data=v)

    def load(self, h5file, name=None):
        if type(h5file) is not h5py.File:
            raise ValueError("Dataset file format must be HDF5.")

        name = name or self.__class__.__name__

        if name in h5file:
            self.data = self.read_h5_recursively(h5file[name])

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

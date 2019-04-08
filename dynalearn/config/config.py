import h5py
import os

class Config:
	def __init__(self, graph_gen, dynamics_gen, model):
		super(Config, self).__init__()
		self.graph_gen = graph_gen
		self.dynamics_gen = dynamics_gen
		self.model = model


    def save_hdf5(self, path):
        data_h5file = h5py.File(os.path.join(path, 'train_data.hdf5'), 'w')
        model_h5file = h5py.File(os.path.join(path, 'model.hdf5'), 'w')

        self.save_data(data_h5file)
        self.save_model(model_h5file)

    def save_hdf5_data(self, file):


    def save_hdf5_model(self, file):



import h5py
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K


class Experiment:
	def __init__(self, name, model, data_generator,
				 loss=binary_crossentropy, optimizer=Adam, 
				 metrics=['accuracy'], learning_rate=1e-4,
				 callbacks=None, numpy_seed=1, tensorflow_seed=2):

		self.name = name
		self.history = {}
		self.epoch = 0

		self.model = model
		self.data_generator = data_generator
		self.graph_model = self.data_generator.graph_model
		self.dynamics_model = self.data_generator.dynamics_model

		self.loss = loss
		self.optimizer = optimizer
		self.optimizer.lr = K.variable(learning_rate)
		self.metrics=metrics
		self.callbacks=callbacks
		self.np_seed = numpy_seed
		self.tf_seed = tensorflow_seed

		self.model.model.compile(self.optimizer, self.loss, self.metrics)


	def generate_data(self, num_sample, T, **kwargs):
		self.data_generator.generate(num_sample, T, **kwargs)

	def train_model(self, epochs, steps_per_epoch, verbose):
		i_epoch = self.epoch
		f_epoch = self.epoch + epochs
		history = self.model.model.fit_generator(self.data_generator,
												 steps_per_epoch=steps_per_epoch,
												 initial_epoch=i_epoch,
												 epochs=f_epoch,
												 verbose=verbose,
												 callbacks=self.callbacks,
												 shuffle=False)

		for k in history.history:
			if k not in self.history:
				self.history[k] = [None] * i_epoch
			elif len(self.history[k]) < i_epoch:
				to_fill = i_epoch - len(self.history[k])
				self.history[k].extend([None] * to_fill)		
			self.history[k].extend(history.history[k])

		self.epoch += epochs

		return history



	def save_hdf5_model(self, h5file):
		model_name = type(self.model).__name__
		model_params = self.model.params
		for name, value in model_params.items():
			h5file.create_dataset('/model/params/' + name, data=value)

		h5file.create_dataset('/model/name', data=model_name)
		weights = self.model.model.get_weights()

		for i, w in enumerate(weights):
			h5file.create_dataset('/model/weights/w_' + str(i), data=w)


	def load_hdf5_model(self, h5file):
		if str(h5file['/model/name/'][...]) != type(self.model).__name__:
			raise ValueError('invalid type for model during loading.')

		weights = [None for i in h5file['/model/weights'].keys()]
		for k, v in h5file['/model/weights'].items():
			weights[int(k[2:])] = np.array(v)

		self.model.model.set_weights(weights)
		return 0


	def save_hdf5_optimizer(self, h5file):
		optimizer_name = type(self.optimizer).__name__
		optimizer_params = self.optimizer.get_config()
		h5file.create_dataset('/optimizer/name', data=optimizer_name)

		for name, value in optimizer_params.items():
			h5file.create_dataset('/optimizer/params/' + name, data=value)

		h5file.create_dataset('/loss', data=self.loss.__name__)
	

	def save_hdf5_history(self, h5file):
		for name, value in self.history.items():
			h5file.create_dataset('/history/' + name, data=value,
								  fillvalue=np.nan)


	def load_hdf5_history(self, h5file):
		for k, v in h5file["/history/"].items():
			self.history[k] = list(v[...])


	def save_hdf5_data(self, h5file):
		graph_name = type(self.graph_model).__name__
		graph_params = self.graph_model.params

		dynamics_name = type(self.dynamics_model).__name__
		dynamics_params = self.dynamics_model.params

		h5file.create_dataset('/graph/name', data=graph_name)
		for name, value in graph_params.items():
			h5file.create_dataset('/graph/params/' + name, data=value)

		h5file.create_dataset('/dynamics/name', data=dynamics_name)
		for name, value in dynamics_params.items():
			h5file.create_dataset('/dynamics/params/' + name, data=value)

		for g_name in self.data_generator.graph_inputs:
			inputs = self.data_generator.state_inputs[g_name]
			adj = self.data_generator.graph_inputs[g_name]
			targets = self.data_generator.targets[g_name]
			h5file.create_dataset('/data/' + g_name + '/adj_matrix', data=adj)
			h5file.create_dataset('/data/' + g_name + '/inputs', data=inputs)
			h5file.create_dataset('/data/' + g_name + '/targets', data=targets)


	def load_hdf5_data(self, h5file):
		for k, v in h5file['/data/'].items():
			self.data_generator.graph_inputs[k] = v["adj_matrix"][...]
			self.data_generator.state_inputs[k] = v["inputs"][...]
			self.data_generator.targets[k] = v["targets"][...]

	def save_hdf5_all(self, h5file):
		h5file.create_dataset('/np_seed', data=self.np_seed)
		h5file.create_dataset('/tf_seed', data=self.tf_seed)
		h5file.create_dataset('/name/', data=self.name)
		self.save_hdf5_model(h5file)
		self.save_hdf5_optimizer(h5file)
		self.save_hdf5_history(h5file)
		self.save_hdf5_data(h5file)

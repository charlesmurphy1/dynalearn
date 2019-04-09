import h5py
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy


class Trainer:
	def __init__(self, model, data_generator,
				 loss=binary_crossentropy, optimizer=Adam, 
				 metrics=['accuracy'], learning_rate=1e-4,
				 callbacks=None, numpy_seed=1, tensorflow_seed=2):
		self.history = {}
		self.epoch = 0

		self.model = model
		self.data_generator = data_generator

		self.loss = loss
		self.optimizer = optimizer(learning_rate)
		self.metrics=metrics
		self.callbacks=callbacks
		self.np_seed = numpy_seed
		self.tf_seed = tensorflow_seed

		np.random.seed(self.np_seed)
		tf.set_random_seed(self.tf_seed)
		self.model.model.compile(self.optimizer, self.loss, self.metrics)


	def generate_data(self, num_sample, T, **kwargs):
		self.data_generator.generate(num_sample, T, **kwargs)

	def train(self, epochs, steps_per_epoch, verbose):
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
		for layer in self.model.model.layers:
			for w, value in zip(layer.weights, layer.get_weights()):
				name = w.name
				h5file.create_dataset('/model/weights/' + name, data=value)


	def save_hdf5_optimizer(self, h5file):
		optimizer_name = type(self.optimizer).__name__
		optimizer_params = self.optimizer.get_config()
		h5file.create_dataset('/optimizer/name', data=optimizer_name)

		for name, value in optimizer_params.items():
			h5file.create_dataset('/optimizer/params/' + name, data=value)

		h5file.create_dataset('/optimizer/loss', data=self.loss.__name__)
		h5file.create_dataset('/optimizer/np_seed', data=self.np_seed)
		h5file.create_dataset('/optimizer/tf_seed', data=self.tf_seed)
	
	def save_hdf5_history(self, h5file):

		for name, value in self.history.items():
			h5file.create_dataset('/history/' + name, data=value,
								  fillvalue=np.nan)

	def save_hdf5_data(self, h5file):
		graph_name = type(self.data_generator.graph_gen).__name__
		graph_params = self.data_generator.graph_gen.params

		dynamics_name = type(self.data_generator.state_gen).__name__
		dynamics_params = self.data_generator.state_gen.params

		h5file.create_dataset('/params/graph/name', data=graph_name)
		for name, value in graph_params.items():
			h5file.create_dataset('/params/graph/' + name, data=value)

		h5file.create_dataset('/params/dynamics/name', data=dynamics_name)
		for name, value in dynamics_params.items():
			h5file.create_dataset('/params/dynamics/' + name, data=value)

		for g_name in self.data_generator.graph_inputs:
			inputs = self.data_generator.state_inputs[g_name]
			adj = self.data_generator.graph_inputs[g_name]
			targets = self.data_generator.targets[g_name]
			h5file.create_dataset('/data/' + g_name + '/adj_matrix', data=adj)
			h5file.create_dataset('/data/' + g_name + '/inputs', data=inputs)
			h5file.create_dataset('/data/' + g_name + '/targets', data=targets)







		
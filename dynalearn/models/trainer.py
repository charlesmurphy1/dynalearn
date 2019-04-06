from tensorflow.keras.optimizer import Adam
from tensorflow.keras.loss import Adam


class Trainer:
	def __init__(self, model, data_generator, loss, optimizer=Adam,
				 learning_rate=1e-4, callbacks=None):
		super(Trainer, self).__init__()
		self.model = model
		self.data_generator = data_generator

		self.loss = loss
		self.optimizer = optimizer(learning_rate)
		self.callbacks = callbacks
		self.epoch = 0

		self.params = {"name": type(self).__name__,
					   "loss": loss.__name__,
					   "optimizer": optimizer.__name__}


	def train(self, epoch, steps_per_epoch)

		
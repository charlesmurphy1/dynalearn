import h5py
import os

class Config:
	def __init__(self, data_gen, model, path):
		super(Config, self).__init__()
        self.graph_name = type(data_gen.graph_gen).__name__
		self.graph_params = data_gen.graph_gen.params
		self.dynamics_name = type(data_gen.state_gen).__name__
        self.dynamics_params = data_gen.state_gen.params
        self.model_name = type(model).__name__
        self.model_params = model.params
        self.optimizer_name = type(model.model.optimizer).__name__
        self.optimizer_params = model.model.optimizer.get_config()
		self.path = path


    def save_hdf5(self, data_generator, model):
        data_h5file = h5py.File(os.path.join(self.path, 'train_data.hdf5'), 'w')
        model_h5file = h5py.File(os.path.join(self.path, 'model.hdf5'), 'w')

        self.save_data(data_generator, data_h5file)
        self.save_model(model, model_h5file)

    def save_hdf5_data(self, data_generator, h5file):
        h5file.create_dataset('/params/graph/name', data=self.graph_name)
        for name, value in self.graph_params.items():
            h5file.create_dataset('/params/graph/' + name, data=value)
        
        h5file.create_dataset('/params/dynamics/name', data=self.dynamics_name)
        for name, value in self.dynamics_params.items():
            h5file.create_dataset('/params/dynamics/' + name, data=value)

        for g_name in data_generator.graph_inputs:
            inputs = data_generator.state_inputs[g_name]
            adj = data_generator.graph_inputs[g_name]
            targets = data_generator.targets[g_name]
            h5file.create_dataset('/data/' + g_name + '/adj_matrix', data=adj)
            h5file.create_dataset('/data/' + g_name + '/inputs', data=inputs)
            h5file.create_dataset('/data/' + g_name + '/targets', data=targets)


    def save_hdf5_model(self, model, file):
        for name, value in self.model_params.items():
            h5file.create_dataset('/model/hyperparams/' + name, data=value)

        h5file.create_dataset('/model/name', data=model_name)
        for layer in model._model.layers:
            for w, value in zip(layer.weights.keys(), layer.get_weights()):
                name = w.name
                h5file.create_dataset('/model/params/' + name, data=value)

        h5file.create_dataset('/optimizer/name', data=optimizer_name)
        for name, value in self.optimizer_params:
            h5file.create_dataset('/optimizer/' + name, data=value)




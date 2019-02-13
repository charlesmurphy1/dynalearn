"""

bm.py

Created by Charles Murphy on 21-08-11.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class General_boltzmann_machine. Within this class,
the sampling algorithms are defined.
"""


import torch
import torch.nn as nn
from math import log, isnan
import numpy as np

from .config import *
from .unit import *
from .param import *
from copy import copy, deepcopy
from ../..utilities.utilities import log_mean_exp, log_std_exp, is_iterable

__all__ = ['General_Boltzmann_Machine']

class General_Boltzmann_Machine(nn.Module):
	"""
	Base class for Boltzmann machine
	
	**Parameters**
	units_info : Dict
		Dict of ``unit.Unit_info`` objects to create unit groups

	params_info : Dict
		Dict of param keys to create parameters. 
	..note::
		Pairs of keys ``"{key1}_{key2}`` create ``param.Weight`` and sigle 
		keys creates ``param.Bias``.

	config : config.Config : (default = ``None``)
		Contains all hyperparameters and paths for General_Boltzmann_Machine 
		objects. If ``None``, it initiliazes a Config object with default
		values.

	..seealso:
		``config.Config

	"""
	def __init__(self, units_info, params_info, config=None):
		super(General_Boltzmann_Machine, self).__init__()

		if config is None:
			config = Config()

		self.units_info = units_info
		self.params_info = params_info
		self.use_cuda = config.USE_CUDA

		self.path_to_model = config.PATH_TO_MODEL
		self.model_name = config.MODEL_NAME
		self.batchsize = config.BATCHSIZE

		self.beta = config.BETA
		self.num_sample = config.NUM_SAMPLE


		self.params = {}
		self.init_params(config.INIT_PARAMS)
		self.init_keys()
		self.mc_units = self.init_units()

		self.log_Z_values = torch.zeros(self.num_sample)
		self.log_Z = 1
		self.log_Z_err = 0
		self.log_Z_to_be_eval = True


	def __str__(self):
		str_out = ""
		for k in self.params:
			str_out += "{0} : {1}\n".format(k,
											self.params[k].param.data.numpy()
											)
		return str_out


	def clone(self):
		return deepcopy(self)



	def forward(self, v_data, num_step):
		if type(v_data) is not dict:
			v_data = {self.v_key: v_data}

		units = self.init_units(v_data)
		units = self.sampler(units, num_step, given=self.v_key)

		data = {}
		for k in self.units_info:
			data[k] = units[k].data
		return data


	# Initialization methods
	def init_units(self, data_dict=None):
		"""
		Initiliazes units dict.

		**Parameters**
		batchsize : Integer
			Size of the batchsize.

		data_dict : Dict : (default = ``None``)
			Dict initializing the units value. If ``None``, the unit values are
			random.

		**Returns**
			units : Dict
			Dict of ``unit.Unit`` object.

		"""
		units = {}
		if data_dict is not None:
			u = next(iter(data_dict.values()))
			batchsize = u.size(0)
		else:
			batchsize = self.batchsize

		for k in self.units_info:
			units[k] = Unit(k, self.units_info[k], batchsize, self.use_cuda)
			if data_dict is not None and k in data_dict:
				units[k].data = data_dict[k].clone()
				if self.use_cuda:
					units[k].data = units[k].data.cuda()



		return units


	def init_params(self, init_params):
		"""
		Initiliazes the Boltzmann machine parameters.

		**Parameters**
		init_params : Dict
			Dict of initialization value. Keys should be in ['w', 'bv', 'bh'].

		"""
		params = {}
		for k in self.params_info:
			if len(k) == 2:
				pair = (self.units_info[k[0]], self.units_info[k[1]])
				params[k] = Weight(pair, init_params["w"], self.use_cuda)
			else:
				unit = self.units_info[k]
				if unit.u_kind == "visible":
					params[k] = Bias(unit, init_params["bv"],
										  self.use_cuda)
				elif unit.u_kind == "hidden":
					params[k] = Bias(unit, init_params["bh"],
										  self.use_cuda)

		self.params = torch.nn.ModuleDict(params)



	def init_keys(self):
		"""
		Initiliazes unit keys according to kind.

		"""
		self.v_key = None
		self.h_keys = set()

		for k, u in self.units_info.items():
			if u.u_kind == "visible":
				if self.v_key is None:
					self.v_key = k
				else:
					raise ValueError("There must be only one set of visible \
									  units")
			if u.u_kind == "hidden":
				self.h_keys.add(k)


	# Energy method
	def energy(self, units):
		"""
		Computes the energy of a unit configuration.

		**Parameters**
		units : Dict
			Dict of ``unit.Unit`` objects.

		**Returns**
		energy : torch.Tensor
			Energy of each element of the minibatch.

		"""
		energy = torch.zeros(units[self.v_key].batchsize)
		if self.use_cuda:
			energy = energy.cuda()

		for k in self.params:
			energy += self.params[k].energy_term(units)

		return energy.mean()

	# Sampling methods
	def free_energy(self, v_data):
		"""
		Computes the free energy of a sample. (virtual)

		**Parameters**
		v_data : torch.Tensor or Dict
			Data of a minibatch. 
		..note:
			For conditional type of BM, it must be a Dict
			whose keys corresponds to visible and condition units.

		**Returns**
		freeEnergy : torch.Tensor
			Free energy of each element of the minibatch.

		"""
		raise NotImplementedError('self.free_energy() has \
								   not been implemented.')
		return 0


	def inference(self, v_data):
		"""
		Computes the posterior given a sample. (virtual)

		**Parameters**
		v_data : torch.Tensor or Dict
			Data of a minibatch. 
		..note:
			For conditional type of BM, it must be a Dict
			whose keys corresponds to visible and condition units.

		**Returns**
		units : Dict
			Dict of ``unit.Unit`` where all hidden unit values have been 
			replaced by their posterior distribution.

		"""
		raise NotImplementedError('self.inference() has \
								   not been implemented.')
		return 0



	def reconstruction(self, v_data):
		"""
		Reconstructs a sample.

		**Parameters**
		v_data : torch.Tensor or Dict
			Data of a minibatch. 
		..note:
			For conditional type of BM, it must be a Dict
			whose keys corresponds to visible and condition units.

		**Returns**
		avg_v : torch.Tensor
			Average activation probability of each visible units.

		"""
		prob, units = self.inference(v_data)

		activation_v = torch.zeros(units[self.v_key].data.size())
		if self.use_cuda:
			activation_v = activation_v.cuda()

		for k in self.params_info:
			if len(k) == 2:
				if k[0] == self.v_key:
					activation_v += self.params[k].mean_term(units, k[1])
				elif k[1] == self.v_key:
					activation_v += self.params[k].mean_term(units, k[0])
			else:
				if k == self.v_key:
					activation_v += self.params[k].mean_term(units, k)

		avg_v = torch.exp(units[self.v_key].log_p(activation_v)).detach()
		return avg_v


	def sampler(self, units, num_step, given=None, with_pcd=False):
		"""
		Samples from the model using MCMC Gibbs sampling. (virtual)

		**Parameters**
		num_step : integer
			Number of Gibbs steps.

		units : Dict
			Dict of ``unit.Unit`` objects from which the Markov chain starts.

		given : Any
			Key of the given unit for the sampling.

		with_pcd : Bool
			With persistent contrastive divergence.

		**Returns**
		units : Dict
			Dict of ``unit.Unit`` objects resulting from the Gibbs sampling.

		"""
		raise NotImplementedError('self.sampler() has \
								   not been implemented.')
		return 0


	def compute_log_Z(self):
		"""
		Computes the log parititon function with annealed importance sampling.

		"""
		if self.log_Z_to_be_eval:
			bm_copy = self.clone()

			p = torch.ones([self.num_sample,
							self.units_info[self.v_key].size]) * 0.5
			units = self.init_units({self.v_key:p})
			log_Z0 = 0

			for u in units.values():
				log_Z0 += u.size * log(2)

			b_prev = self.beta[0]

			log_w = torch.zeros(self.num_sample)
			if self.use_cuda:
				log_w = log_w.cuda()

			for i, b in enumerate(self.beta[1:]):
				for k in self.params:
					bm_copy.params[k].param.data = b * self.params[k].param.data
				units = bm_copy.sampler(units, 1, self.v_key, with_pcd=False)
				log_w -= (b - b_prev) * self.energy(units).detach().numpy()
				b_prev = b

			self.log_Z_values = log_w + log_Z0

			self.log_Z = log_mean_exp(self.log_Z_values.numpy())
			self.log_Z_err = np.std(self.log_Z_values.numpy())
			self.log_Z_to_be_eval = False


	def log_likelihood(self, v_data):
		"""
		Computes the log likelihood.

		**Parameters**
		v_data : torch.Tensor or Dict
			Data of a minibatch. 

		**Returns**
		log_p : torch.Tensor

		"""

		if self.log_Z_to_be_eval:
			self.compute_log_Z()

		log_Z = log_mean_exp(self.log_Z_values)
		free_energy = self.free_energy(v_data).detach()
		log_p = - free_energy - log_Z

		return log_p


	def save_params(self):
		"""
		Saves the parameters to the model config file.

		"""
		from os.path import join
		
		# always saved in cpu mode
		path = self.path_to_model
		name = self.model_name
		params = {}
		for k in self.params:
			params[k] = self.params[k].param.data.cpu()

		torch.save(params, join(path, name + ".pt"))

		return 0


	def load_params(self, fileloc):
		"""
		Loads the parameters from the model config file.

		"""

		params = torch.load(fileloc, map_location="cpu")

		# convert to cuda if use_cuda
		for k in self.params:
			self.params[k].param.data = params[k].clone()
			
			if self.use_cuda:
				self.params = self.params.cuda()

		return 0

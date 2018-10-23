"""

bm.py

Created by Charles Murphy on 21-08-11.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class General_boltzmann_machine. Within this class,
the sampling algorithms are defined.
"""


import torch
from .unit import *
from .param import *
from copy import copy
from utilities.utilities import log_mean_exp, log_std_exp
from math import log
import numpy as np

__all__ = ['General_Boltzmann_Machine']

class General_Boltzmann_Machine(object):
	"""
	Base class for Boltzmann machine
	
	**Parameters**
	units_info : Dict
		Dict of ``unit.Unit_info`` objects to create unit groups

	params_info : Dict
		Dict of param keys to create paramters. 
	..note::
		Pairs of keys ``tuple(key1, key2)`` create ``param.Weight`` and sigle 
		keys creates ``param.Bias``.

	model_config : config.Model_Config : (default = ``None``)
		Contains all hyperparameters and paths for General_Boltzmann_Machine 
		objects. If ``None``, it initiliazes a Model_Config object with default
		values.

	..seealso:
		``config.Model_Config

	"""
	def __init__(self, units_info, params_info, model_config=None):
		super(General_Boltzmann_Machine, self).__init__()

		if model_config is None:
			model_config = BM_config()

		self.units_info = units_info
		self.params_info = params_info
		self.model_config = model_config
		self.use_cuda = model_config.USE_CUDA

		self.params = {}
		self.mc_units = self.init_units(model_config.BATCHSIZE)
		self.init_keys()
		self.init_params(model_config.INIT_PARAMS)

		self.log_Z_values = torch.zeros(self.model_config.NUM_SAMPLE)


	def __str__(self):
		str_out = ""
		for k in self.params:
			str_out += "{0} : {1}\n".format(k, self.params[k].value.numpy())
		return str_out


	def __copy__(self):
		bm = General_Boltzmann_Machine(self.units_info, self.params_info,
									   0.01, None, self.use_cuda)

		for k in bm.params:
			bm.params[k].value = self.params[k].value.clone()

		return bm


	# Initialization methods
	def init_units(self, batchsize, value_dict=None):
		"""
		Initiliazes units dict.

		**Parameters**
		batchsize : Integer
			Size of the batchsize.

		value_dict : Dict : (default = ``None``)
			Dict initializing the units value. If ``None``, the unit values are
			random.

		**Returns**
			units : Dict
			Dict of ``unit.Unit`` object.

		"""
		units = {}
		if value_dict is not None:
			u = next(iter(value_dict.values()))
			batchsize = u.size(0)

		for k in self.units_info:
			units[k] = Unit(k, self.units_info[k], batchsize, self.use_cuda)
			if value_dict is not None and k in value_dict:
				units[k].value = value_dict[k]

		return units


	def init_params(self, init_params):
		"""
		Initiliazes the Boltzmann machine parameters.

		**Parameters**
		init_params : Dict
			Dict of initialization value. Keys should be in ['w', 'bv', 'bh'].

		"""
		for k in self.params_info:
			if util.is_iterable(k):
				unit_pair = (self.units_info[k[0]], self.units_info[k[1]])
				self.params[k] = Weight(pair, init_params["w"])
			else:
				unit = self.units_info[k]
				if unit.u_kind == "visible":
					self.params[k] = Bias(unit, init_params["bv"])
				elif unit.u_kind == "hidden":
					self.params[k] = Bias(unit, init_params["bh"])



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
					raise ValueError("There must be only one set of visible units")
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

		for k in self.params:
			energy += self.params[k].energy_term(units)

		return energy

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
		post_dis : torch.Tensor
			Posterior given each sample of the minibatch.

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
		units = self.inference(v_data)


		activation_v = torch.zeros(units[self.v_key].value.size())
		for k1, k2 in self.params_info:
			if k1 == self.v_key:
				activation_v += self.params[(k1, k2)].mean_term(units, k2)
			elif k2 == self.v_key:
				activation_v += self.params[(k1, k2)].mean_term(units, k1)
		avg_v = util.sigmoid(activation_v)
		return avg_v

	def conditional_log_p(self, units):
		"""
		Computes the conditonal log-probability given the current configuration.
		(virtual)

		**Parameters**
		units : Dict
			Dict of ``unit.Unit`` objects.

		**Returns**
		avg_v : torch.Tensor
			Conditonal log-probability.

		"""
		raise NotImplementedError('self.conditional_prob() has \
								   not been implemented.')
		return 0


	def sampler(self, num_step, units, given="v"):
		"""
		Samples from the model using MCMC Gibbs sampling. (virtual)

		**Parameters**
		num_step : integer
			Number of Gibbs steps.

		units : Dict
			Dict of ``unit.Unit`` objects from which the Markov chain starts.

		**Returns**
		units : Dict
			Dict of ``unit.Unit`` objects resulting from the Gibbs sampling.

		"""
		raise NotImplementedError('self.sampler() has \
								   not been implemented.')
		return 0


	# Learning methods
	def positive_phase(self, units):

		units = self.inference(v)

		pos_phase = {}

		for k in self.params:
			pos_phase[k] = self.params[k].phase(units)

		return pos_phase


	def negative_phase(self, units, num_step):
		if units is None:
			units = self.mc_units
		units = self.sampler(units, num_step)
		neg_phase = {}

		for k in self.params:
			neg_phase[k] = self.params[k].phase(units)

		return neg_phase




	def compute_log_Z(self):

		beta = self.model_config.BETA
		num_sample = self.model_config.NUM_SAMPLE

		bm_copy = copy(self)
		p = torch.ones([num_sample, self.units_info[self.v_key].size]) * 0.5
		units = self.init_units({self.v_key:p})
		log_Z0 = 0

		for u in units.values():

			log_Z0 += u.size * log(2)
		b_prev = beta[0]
		log_w = torch.zeros(num_sample)

		for i, b in enumerate(beta[1:]):

			for k in self.params:
				bm_copy.params[k].value = b * self.params[k].value
			units = bm_copy.sampler(units, 1, self.v_key)
			log_w -= (b - b_prev) * self.energy(units)
			b_prev = b

		self.log_Z_values = log_w + log_Z0


	def log_likelihood(self, v_data):

		# Likelihood calculation
		log_Z = util.log_mean_exp(self.log_Z_values)
		free_energy = self.free_energy(v_data)
		log_p = - free_energy - log_Z

		return log_p


	def reconstruction_MSE(self, v_data):

		# Mean square error on reconstruction
		mean_recon = self.reconstruction(v_data)
		if util.is_iterable(v_data):
			v = v_data[self.v_key]
		else:
			v = v_data

		recon_mse = (v - mean_recon)**2

		return torch.mean(recon_mse, 1)


	def update_params(self, grad, lr, w=0):

		for k in self.params:
			self.params[k].value += lr * grad[k]
			if type(k) is tuple and w > 0:
				self.params[k].value -= lr * w * self.params[k].value

		return 0


	def save_params(self):
		from os.path import join
		
		# always saved in cpu mode
		path = self.model_config.PATH_TO_MODEL
		name = self.model_config.MODEL_NAME
		params = {}
		for k in self.params:
			params[k] = self.params[k].value.cpu()

		torch.save(params, join(path, name + ".pt"))

		return 0


	def load_params(self, fileloc):

		params = torch.load(fileloc, map_location="cpu")

		# convert to cuda if use_cuda
		for k in self.params:
			self.params[k].value = params[k].clone()
			
			if self.use_cuda:
				self.params = self.params.cuda()

		return 0



if __name__ == '__main__':

	units_info = {"v": Unit_info("v", 3, "visible"),
				  "h1": Unit_info("h1", 5, "hidden"),
				  "h2": Unit_info("h2", 4, "hidden"),}
	params_info = [("v", "h1"), ("v", "h2")]
	batchsize = 1

	bm = General_Boltzmann_Machine(units_info, params_info)

	print("Weight size: ", bm.params[("v", "h1")].size())
	print("Weight size: ", bm.params[("v", "h2")].size())
	print("Energy of config: ", bm.energy(bm.mc_units))
	print("Visible key: ", bm.v_key)
	print("Hidden keys: ", bm.h_keys)


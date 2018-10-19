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
	"""docstring for General_Boltzmann_Machine"""
	def __init__(self, units_info, weights_info, with_bias=True,
				 init_scale=0.01, p=None, use_cuda=False):
		super(General_Boltzmann_Machine, self).__init__()
		self.units_info = units_info
		self.weights_info = weights_info
		self.with_bias = with_bias
		self.use_cuda = use_cuda

		self.params = {}
		self.mc_units = self.init_units()
		self.init_keys()
		self.init_weights(init_scale)
		self.init_biases(p)


	def __repr__(self):
		return "<bm.General_Boltzmann_Machine>"


	def __str__(self):
		str_out = ""
		for k in self.params:
			str_out += "{0} : {1}\n".format(k, self.params[k].value.numpy())
		return str_out


	def __copy__(self):
		bm = General_Boltzmann_Machine(self.units_info, self.weights_info,
									   0.01, None, self.use_cuda)

		for k in bm.params:
			bm.params[k].value = self.params[k].value.clone()

		return bm


	# Initialization methods
	def init_units(self, value_dict=None, batchsize=32):
		units = {}
		if value_dict is not None:
			u = next(iter(value_dict.values()))
			batchsize = u.size(0)

		for k in self.units_info:
			units[k] = Unit(k, self.units_info[k], batchsize, self.use_cuda)
			if value_dict is not None and k in value_dict:
				units[k].value = value_dict[k]

		return units


	def init_weights(self, init_scale=0.01):

		for k in self.weights_info:
			pair = (self.mc_units[k[0]], self.mc_units[k[1]])
			self.params[k] = Weight(pair, init_scale)


	def init_biases(self, p=None):

		if p is None:
			p = {}
			for k in self.units_info:
				p[k] = None
		# biases
		if self.with_bias:
			for k in self.units_info:
				kind = self.mc_units[k].u_kind
				self.params[k] = Bias(self.mc_units[k], p[k])


	def init_keys(self):
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
		energy = torch.zeros(units[self.v_key].batchsize)

		for k in self.params:
			energy += self.params[k].energy_term(units)

		return energy

	# Sampling methods
	def free_energy(self, v):
		raise NotImplementedError('self.free_energy() has not been implemented.')
		return 0


	def inference(self, v):
		raise NotImplementedError('self.inference() has not been implemented.')
		return 0



	def reconstruction(self, v):
		p_inference = self.inference(v)
		p_inference[self.v_key] = v
		units = self.init_units(p_inference)

		activation_v = torch.zeros(units[self.v_key].value.size())
		for k1, k2 in self.weights_info:
			if k1 == self.v_key:
				activation_v += self.params[(k1, k2)].mean_term(units, k2)
			elif k2 == self.v_key:
				activation_v += self.params[(k1, k2)].mean_term(units, k1)

		return util.sigmoid(activation_v)

	def conditional_log_p(self, v):
		raise NotImplementedError('self.conditional_prob() has not been implemented.')
		return 0




	def sampler(self, units, num_step=1, given="v"):
		raise NotImplementedError('self.sampler() has not been implemented.')
		return 0


	# Learning methods
	def positive_phase(self, v):

		u_inference = self.inference(v)
		u_inference[self.v_key] = v
		units = self.init_units(u_inference)

		pos_phase = {}

		for k in self.params:
			pos_phase[k] = self.params[k].phase(units)

		return pos_phase


	def negative_phase(self, units=None, num_step=1):
		if units is None:
			units = self.mc_units
		units = self.sampler(units, num_step)
		neg_phase = {}

		for k in self.params:
			neg_phase[k] = self.params[k].phase(units)

		return neg_phase




	def _log_Z(self, num_sample=10, betas=None, recompute=True):

		if betas is None:
			betas = np.linspace(0, 0.5, 100)
			betas = np.append(betas,np.linspace(0.5, 0.9, 800))
			betas = np.append(betas,np.linspace(0.9, 1., 2000))

		if recompute:
			bm_copy = copy(self)
			p = torch.ones([num_sample, self.units_info[self.v_key].size]) * 0.5
			units = self.init_units({self.v_key:p})
			log_Z0 = 0

			for u in units.values():

				log_Z0 += u.size * log(2)
			b_prev = betas[0]
			log_w = torch.zeros(num_sample)

			for i, b in enumerate(betas[1:]):

				for k in self.params:
					bm_copy.params[k].value = b * self.params[k].value
				units = bm_copy.sampler(units, 1, self.v_key)
				log_w -= (b - b_prev) * self.energy(units)
				b_prev = b

			self.log_Z_values = log_w + log_Z0

		log_mean = log_mean_exp(self.log_Z_values)

		return log_mean


	def log_likelihood(self, v, num_sample=10, betas=None, recompute=False):

		# Likelihood calculation
		log_Z = self._log_Z(num_sample, betas, recompute)
		free_energy = self.free_energy(v)
		log_p = - free_energy - log_Z
		# print(free_energy, log_Z)

		return log_p


	def reconstruction_MSE(self, v):

		# Mean square error on reconstruction
		mean_recon = self.reconstruction(v)
		recon_mse = (v - mean_recon)**2

		return torch.mean(recon_mse, 1)


	def update_params(self, grad, lr, weight_decay=0):

		for k in self.params:
			self.params[k].value += lr * grad[k]
			if type(k) is tuple and weight_decay > 0:
				self.params[k].value -= lr * weight_decay * self.params[k].value

		return 0


	def save_params(self, filename):
		
		# always saved in cpu mode

		params = {}
		for k in self.params:
			params[k] = self.params[k].value.cpu()

		torch.save(params, filename)

		return 0


	def load_params(self, filename):

		params = torch.load(filename, map_location="cpu")

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
	weights_info = [("v", "h1"), ("v", "h2")]
	batchsize = 1

	bm = General_Boltzmann_Machine(units_info, weights_info, batchsize, init_scale=0.01, p=None)

	print("Weight size: ", bm.params[("v", "h1")].size())
	print("Weight size: ", bm.params[("v", "h2")].size())
	print("Energy of config: ", bm.energy(bm.mc_units))
	print("Visible key: ", bm.v_key)
	print("Hidden keys: ", bm.h_keys)


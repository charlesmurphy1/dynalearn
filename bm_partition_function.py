"""

bm_partition_function.py

Created by Charles Murphy on 01-08-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class BM_partition_function. This class generate procedures to compute
an approximation of a BM partition function.
"""

import torch
import utilities as util
from math import log, exp
from unit import *
from param import *
from copy import copy

class BM_partition_function(object):
	"""docstring for BM_partition_function"""
	def __init__(self, bm):

		self.bm = bm
		self.bm_eval = copy(bm)
		
		self.batchsize = self.bm.batchsize

		self.log_Z0 = 0

		for u in self.bm.init_units().values():
			print(u.size, u.value)
			self.log_Z0 += u.size * log(2)

		# print(2**self.log_Z0)

		super().__init__()
		

	def get_sample(self, units, beta):

		for k in self.bm.params:
			self.bm_eval.params[k].value = beta * self.bm.params[k].value

		return self.bm_eval.sampler(units, 1, "v")


	def log_approximate(self, betas):

		units = self.bm.init_units()
		# act = self.base_bias_v.activation_term(units, None)
		p = torch.ones(self.batchsize, units["v"].size) * 0.5
		units["v"].value = util.random_binary(p)

		dH = torch.zeros(self.batchsize, len(betas) - 1)
		db = torch.Tensor([betas[i] - betas[i - 1] for i in range(1, len(betas))])

		for i, b in enumerate(betas[1:]):
			# print(i, b)
			units = self.get_sample(units, b)
			dH[:, i] = - self.bm.energy(units)

		log_w = torch.matmul(dH, db)
		a = torch.max(log_w)

		return self.log_Z0 + a + torch.log(torch.exp(log_w - a).mean())

if __name__ == '__main__':
	import numpy as np
	from math import log2, floor
	from rbm import RBM
	from unit import *
	from utilities import *


	def exact_rbm_logPF(rbm):
		N = rbm.mc_units["v"].size 
		M = rbm.mc_units["h"].size

		rbm_copy = RBM(N, M, 1)

		for k in rbm_copy.params:
			rbm_copy.params[k].value = rbm.params[k].value.clone()

		units = rbm.init_units()
		units["v"].value = torch.zeros(1, N)
		units["h"].value = torch.zeros(1, M)

		Z = 0
		for i in range(2**N):
			for j in range(2**M):
				H = rbm_copy.energy(units)
				# print((i,"/",2**N, ", ", j, "/",2**M), units["v"].value, units["h"].value, H)
				Z += torch.exp(-H)
				units["h"].value[0,:] = add_one_to_bits_torch(units["h"].value[0,:])

			units["v"].value[0,:] = add_one_to_bits_torch(units["v"].value[0,:])

		return np.log(Z[0])

	batchsize = 5
	N = 1
	M = 3

	use_cuda = False
	rbm = RBM(N, M, batchsize)

	rbm.params["v"].value.fill_(3)
	rbm.params["h"].value.fill_(2)
	rbm.params[("v", "h")].value.fill_(1)

	# rbm.params["v"].value.normal_()
	# rbm.params["h"].value.normal_()
	# rbm.params[("v", "h")].value.normal_()


	betas = np.array([])
	betas = np.append(betas, np.linspace(0., 0.5, 500))
	betas = np.append(betas, np.linspace(0.5, 0.9, 4000))
	betas = np.append(betas, np.linspace(0.9, 1., 10000))

	partfunc = BM_partition_function(rbm)
	log_Z_approx = partfunc.log_approximate(betas)
	log_Z_exact = exact_rbm_logPF(rbm)

	print("Appox: ", log_Z_approx)
	print("Exact: ", log_Z_exact)
	print("Ratio: ", np.exp(float(log_Z_exact) - float(log_Z_approx)))


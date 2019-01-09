from torchvision.datasets import MNIST
import torch
import progressbar
import os
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from dynalearn.model.rbm import *
from dynalearn.model.config import *
from dynalearn.trainer.bm_trainer import *
from dynalearn.trainer.history import *
from dynalearn.trainer.bm_statistics import *

def load_data(num_train=-1, num_val=1, numbers=-1):
	dataset = MNIST("testdata/mnist", download=True)

	if numbers == -1:
		numbers = list(range(10))
	elif type(numbers) is int:
		numbers = [numbers]

	dataset = list(dataset)
	shuffle(dataset)
	if num_train > (len(dataset) - num_val) or num_train == -1:
		num_train = len(dataset) - num_val
	train_dataset = []
	train_label = []
	val_dataset = []
	val_label = []


	for i, d in enumerate(dataset):
		if d[1] in numbers:
			data = np.array(d[0], dtype=float)
			data.resize(data.size)
			data[data>0] = 1
			data[data<=0] = 0
			data = torch.Tensor(data)

			if len(train_dataset) < num_train:
				train_dataset.append(data)
				train_label.append(int(d[1]))
			else:
				if len(val_dataset) < num_val:
					val_dataset.append(data)
					val_label.append(int(d[1]))
				else: break


	p_train = None
	for data in train_dataset:
		if p_train is None:
			p_train = torch.zeros(len(data))
		p_train += data / len(train_dataset)


	return train_dataset, train_label, val_dataset, val_label, p_train




def setup_history(config, bm,
				  with_param=True,
				  with_grad=False,
				  with_logp=False,
				  with_partfunc=False,
				  with_pseudo=True,
				  with_free_energy=True,
				  with_recon=True):
	graining = config.GRAINING
	makeplot = config.MAKEPLOT
	statstics = {}
	if with_param:
		for k in bm.params:
			statstics[("param", k)] = Parameter_Statistics(k, makeplot=makeplot)

	if with_grad:
		for k in bm.params:
			statstics[("grad", k)] = Gradient_Statistics(k, makeplot=makeplot)

	if with_logp:
		statstics["log-p"] = LogLikelihood_Statistics(graining=graining,
													  makeplot=makeplot)

	if with_partfunc:
		statstics["partfunc"] = Partition_Function_Statistics(makeplot=makeplot)


	if with_free_energy:
		statstics["free_energy"] = Free_Energies_Statistics(graining=graining,
															makeplot=makeplot)

	if with_recon:
		statstics["recon"] = Reconstruction_MSE_Statistics(graining=graining,
														   makeplot=makeplot)


	criterion = Reconstruction_MSE_Statistics(graining=graining,
                                              makeplot=False)
	# criterion = LogLikelihood_Statistics(graining=graining,
	# 									 makeplot=False)
	return History("minst_history", statstics, criterion,
				   config.PATH_TO_HISTORY)


def plot_number(imag, ax):
	ax.imshow(imag, cmap="Greys")
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	ax.get_xaxis().set_ticklabels([])
	ax.get_yaxis().set_ticklabels([])
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)


def test_rbm(rbm, config, examples, steps=10, intermediate=10):

	fig, ax = plt.subplots(steps + 1, len(examples) + 1)

	for i, l in enumerate(examples):
		e = examples[l]
		units = rbm.init_units({"v" : e})
		num_steps = 0

		imag = np.resize(e, [28, 28])
		plot_number(imag, ax[0, i])
		if i == 0:
			ax[0, i].set_ylabel("0")

		for j in range(1, steps + 1):


			num_steps = intermediate**j - num_steps
			units = rbm.sampler(units, num_steps, given="v")

			imag = np.resize(units["v"].data.numpy(), [28, 28])
			plot_number(imag, ax[j, i])
			if i == 0:
				ax[j, i].set_ylabel(str(intermediate**(j-1)))

	units = rbm.init_units()
	imag = np.resize(units["v"].data.numpy(), [28, 28]) 
	plot_number(imag, ax[0, -1])
	num_steps = 0
	for j in range(1, steps + 1):

		num_steps = intermediate**j - num_steps
		units = rbm.sampler(units, num_steps, given="v")

		imag = np.resize(units["v"].data.numpy(), [28, 28]) 
		plot_number(imag, ax[j, -1])
		

	figname = os.path.join(config.RUN, "test_from_model.png")
	fig.savefig(figname)
	plt.show()





def main():

	# Loading data
	normalize = False
	verbose = True
	num_train = 2
	num_val = 100
	# numbers = [6]
	numbers = list(range(10))
	# numbers = [8]
	train_d, train_l, val_d, val_l, p_train = load_data(num_train, num_val,
														numbers=numbers)

	# # Making RBM
	n_visible = 28 * 28 # number of pixels in MNIST examples
	n_hidden = 128
	batchsize = 16
	lr = 1e-2
	wd = 1e-3
	momentum = 0.8
	numsteps = 5
	numepochs = 200

	show_example = True

	if show_example:
		for d in train_d:
			fig, ax = plt.subplots(1, 1)
			d = np.resize(d, [28, 28])
			plot_number(d, ax)
			plt.show()

	# p = np.resize(p, [28, 28])
	# plt.imshow(p)
	# plt.show()

	config = Config(# Model config
					run_name="testdata/overfit_run",
					model_name='mnist_model',
					batchsize=batchsize,
                	# Training config
					lr=lr, wd=wd, momentum=momentum, numsteps=numsteps, 
					numepochs=numepochs, with_pcd=True, bv_init=0.5,
					makeplot=True, path_to_history='history',
					graining=1.0, keepbest=False, overwrite=False,
                 	)
	# config.save()


	rbm = RBM_BernoulliBernoulli(n_visible, n_hidden, config)
	history = setup_history(config, rbm,
							with_param=False,
							with_grad=True,
							with_logp=False,
							with_partfunc=False,
							with_free_energy=True,
							with_recon=True)
	print("Training phase: {} examples\n---------------".format(len(train_d)))
	trainer = BM_trainer(rbm, history, config)
	trainer.train(train_d, val_d)
	history.make_plots(save=True, show=False, showbest=True)
	

	print("Testing phase\n-------------")
	rbm.load_params(os.path.join(config.PATH_TO_MODEL, config.MODEL_NAME+".pt"))
	
	numbers = list(range(10))

	examples = {}
	i = 0
	for d, l in zip(val_d, val_l):
		if l == numbers[i]:
			examples[l] = torch.Tensor(d)
			examples[l] = torch.reshape(examples[l], [1, examples[l].size(0)])
			# numbers.remove(l)
			i += 1


		if i == len(numbers):
			break

	test_rbm(rbm, config, examples, 10, 2)


	bv = rbm.params["v"].param.data.detach().clone()
	bv = bv.numpy()
	bv = np.reshape(bv, [28,28])

	plt.imshow(bv)
	plt.colorbar()
	plt.show()

if __name__ == '__main__':
	main()

from torchvision.datasets import MNIST
import torch
import progressbar
import os
import numpy as np
import matplotlib.pyplot as plt

from dynalearn.model.rbm import *
from dynalearn.model.config import *
from dynalearn.trainer.bm_trainer import *
from dynalearn.trainer.history import *
from dynalearn.trainer.bm_statistics import *

def load_data(num_data=-1, verbose=False, normalize=False, mean=0., scale=1., numbers=-1):
	dataset = MNIST("testdata/mnist", download=True)

	if numbers == -1:
		numbers = list(range(10))
	elif type(numbers) is int:
		numbers = [numbers]


	if num_data > len(dataset) or num_data == -1:
		num_data = len(dataset)
	formated_dataset = []
	labels = []

	if verbose:
		widgets = [ "Loading data: ",
					progressbar.Bar('-'), ' ',
					progressbar.Percentage(), ' ',
					progressbar.ETA()]
		bar = progressbar.ProgressBar(widgets=widgets, maxval=num_data).start()

	for i, d in enumerate(dataset):
		if d[1] in numbers:
			data = np.array(d[0], dtype=float)
			data.resize(data.size)
			data[data>0] = 1
			data[data<=0] = 0
			data = torch.Tensor(data)
			labels.append(int(d[1]))

			# formated_dataset.append([data, label])
			formated_dataset.append(data)

			if verbose:
				bar.update(i)

			if len(formated_dataset) == num_data:
				break

	if verbose:
		bar.finish()

	if normalize:
		if mean == 0. and scale == 1.:
			if verbose:
				widgets = [ "Normalizing: ",
							progressbar.Bar('-'), ' ',
							progressbar.Percentage(), ' ',
							progressbar.ETA()]
				bar = progressbar.ProgressBar(widgets=widgets, maxval=num_data).start()
			mean = 0
			sqmean = 0
			for i, d in enumerate(formated_dataset):
				mean += torch.mean(d) / len(formated_dataset)
				sqmean += torch.mean(d**2) / len(formated_dataset)
				if verbose:
					bar.update(i)


			scale = torch.sqrt(sqmean - mean**2)
			if verbose:
				bar.finish()

		for i, d in enumerate(formated_dataset):
			formated_dataset[i] = (d - mean) / scale


	return formated_dataset, labels, mean, scale 


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


def test_rbm(rbm, examples, steps=10, intermediate=10):

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
		

	figname = os.path.join("./testdata/", "test_from_model.png")
	fig.savefig(figname)
	# plt.show()





def main():

	# Loading data
	normalize = False
	verbose = True
	n_data = 60000
	# numbers = [6]
	numbers = list(range(10))
	numbers = [8]
	dataset, labels, mean, scale = load_data(n_data, False, normalize, 
											 numbers=numbers)

	# Making RBM
	n_visible = 28 * 28 # number of pixels in MNIST examples
	n_hidden = 500
	batchsize = 64
	lr = 1e-4
	wd = 0
	momentum = 0
	val_size = 0.1
	numsteps = 10
	numepochs = 50

	config = Config(# Model config
					run_name="testdata/run",
					model_name='mnist_model',
					batchsize=batchsize,
                	# Training config
					lr=lr, wd=wd, momentum=momentum, val_size=val_size,
					numsteps=numsteps, numepochs=numepochs, with_pcd=True,
					makeplot=True, path_to_history='mnist_history',
					graining=0.05,
                 	)


	rbm = RBM_BernoulliBernoulli(n_visible, n_hidden, config)
	history = setup_history(config, rbm,
							with_param=True,
							with_grad=False,
							with_logp=True,
							with_partfunc=True,
							with_free_energy=True,
							with_recon=True)
	print("Training phase: {} examples\n---------------".format(len(dataset)))
	trainer = BM_trainer(rbm, history, config)
	trainer.train(dataset)
	history.make_plots(save=True, show=False, showbest=True)
	history.save()
	config.save()
	rbm.save_params()

	print("Testing phase\n-------------")
	rbm.load_params(os.path.join(config.PATH_TO_MODEL, config.MODEL_NAME+".pt"))
	
	numbers = list(range(10))
	dataset, labels, mean, scale = load_data(n_data, False, normalize, 
											 numbers=numbers)
	examples = {}
	i = 0
	for d, l in zip(dataset, labels):
		if l == numbers[i]:
			examples[l] = torch.Tensor(d) * scale + mean
			examples[l] = torch.reshape(examples[l], [1, examples[l].size(0)])
			# numbers.remove(l)
			i += 1


		if i == len(numbers):
			break

	test_rbm(rbm, examples, 10, 2)

if __name__ == '__main__':
	main()

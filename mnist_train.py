from torchvision.datasets import MNIST
import progressbar
import numpy as np
from crbm import *
from rbm import *
from statistics import *
from history import *
from bm_trainer import *

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

def train(bm, path, dataset, n_epoch=10, patience=2, batchsize=32,
		  lr_min=1e-6, lr_max=1e-3, weight_decay=1e-4, momentum=0.,
		  save=False,
		  show=False,
		  num_steps=1,
		  val_prop = 0.08,
		  eval_step=1,
		  with_param=True,
		  with_grad=True,
		  with_logp=True,
		  with_partfunc=True,
		  with_pseudo=True,
		  with_free_energy=True,
		  with_recon=True):

	statstics = {}
	eval_step = {}
	if with_param:
		for k in bm.params:
			statstics[("param", k)] = Parameter_Statistics(k, makeplot=True)
			eval_step[("param", k)] = 1

	if with_grad:
		for k in bm.params:
			statstics[("grad", k)] = Gradient_Statistics(k, makeplot=True)
			eval_step[("grad", k)] = 1

	if with_logp:
		statstics["log-p"] = LogLikelihood_Statistics(recompute=False,
													  graining=50,
													  makeplot=True)
		eval_step["log-p"] = "update"

	if with_partfunc:
		statstics["partfunc"] = Parition_Function_Statistics(makeplot=True)
		eval_step["partfunc"] = 1

	if with_pseudo:
		statstics["pseudo"] = Pseudolikelihood_Statistics(graining=50,
														  makeplot=True)
		eval_step["pseudo"] = "update"

	if with_free_energy:
		statstics["free_energy"] = Free_Energies_Statistics(graining=50,
															makeplot=True)
		eval_step["free_energy"] = "update"

	if with_recon:
		statstics["recon"] = Reconstruction_MSE_Statistics(graining=50,
														   makeplot=True)
		eval_step["recon"] = "update"

	criterion = Reconstruction_MSE_Statistics(graining=50,
                                              makeplot=False)
	history = BM_History(statstics, criterion, path)
	bm_trainer = BM_trainer(bm, "minst_rbm", history=history,
							weight_decay=weight_decay,
							momentum=momentum)

	lr = [lr_max / (i + 1) + (i + 1) / n_epoch * lr_min for i in range(n_epoch)]

	bm_trainer.train(dataset, val_dataset=None, n_epoch=n_epoch, patience=patience,
					 batchsize=batchsize, keep_best=True, lr=lr, num_steps=num_steps,
					 val_prop=val_prop, eval_step=eval_step, save=save, show=show,
					 path=path, verbose=True)

	return 0


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

	print(steps, len(examples))
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

			imag = np.resize(units["v"].value.numpy(), [28, 28])
			plot_number(imag, ax[j, i])
			if i == 0:
				ax[j, i].set_ylabel(str(intermediate**(j-1)))

	units = rbm.init_units()
	imag = np.resize(units["v"].value.numpy(), [28, 28]) 
	plot_number(imag, ax[0, -1])
	num_steps = 0
	for j in range(1, steps + 1):

		num_steps = intermediate**j - num_steps
		units = rbm.sampler(units, num_steps, given="v")

		imag = np.resize(units["v"].value.numpy(), [28, 28]) 
		plot_number(imag, ax[j, -1])
		

	fig.savefig("./mnist_test/test_from_model.png")
	plt.show()





def main():

	# Loading data
	normalize = False
	verbose = True
	n_data = 60000
	n_epoch = 50
	patience = 3
	numbers = -1
	dataset, labels, mean, scale = load_data(n_data, False, normalize, numbers=numbers)

	# Making RBM
	n_visible = 28 * 28 # number of pixels in MNIST examples
	n_hidden = 150
	init_scale = 0.01
	p = None
	batchsize = 16
	use_cuda = True

	rbm = RBM(n_visible, n_hidden,
			  v_kind="bernoulli",
			  init_scale=init_scale,
			  p=p,
			  use_cuda=use_cuda)
	print("Training phase: {} examples\n---------------".format(len(dataset)))
	train(rbm, "./mnist_test/", dataset, n_epoch=n_epoch, patience=patience, 
		  batchsize=batchsize, lr_min=1e-3, lr_max=1e-2, weight_decay=1e-3, 
		  momentum=0.9,
		  save=True, show=False,
		  num_steps=10,
		  val_prop=0.01,
		  eval_step=1,
		  with_param=True,
		  with_grad=False,
		  with_logp=True,
		  with_partfunc=True,
		  with_pseudo=True,
		  with_free_energy=True,
		  with_recon=True)

	print("Testing phase\n-------------")

	rbm.load_params("./mnist_test/best_minst_rbm.pt")

	if numbers == -1:
		numbers = list(range(10))
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

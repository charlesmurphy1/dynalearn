"""

utilities.py

Created by Charles Murphy on 21-08-11.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines a variety of useful functions for bm use and training.
"""

import torch
import numpy as np
from scipy.interpolate import interp1d

def sigmoid(x):
	return 1 / (1 + torch.exp(-x))


def random_binary(p, use_cuda=False):
	dim = p.size()
	r = torch.randn(dim)

	ans = torch.zeros(dim)
	if use_cuda:
		r = r.cuda()

	ans[r<p] = 1.
	ans[r>=p] = 0.

	return ans


def is_iterable(x):
	try:
		iter(x)
		return True
	except:
		return False


def get_bits(x, size=None):

	if size is None or size < floor(log2(x)) + 1:
		size = floor(log2(x)) + 1


	return (x // 2**np.arange(size)) % 2


def add_one_to_bits(x):

	val = x * 1
	i = np.where(x==0)[0][0]
	index = np.arange(len(x))
	val[index <= i] = 0
	val[i] = 1

	return val

def add_one_to_bits_torch(x):
	# print(np.where(x.numpy()==0))
	if torch.all(x==1):
		val = torch.zeros(len(x))
	else:
		i = int(np.where(x.numpy()==0)[0][0])

		index = torch.arange(len(x))
		zeros = torch.zeros(len(x))
		# print(x, i, index)
		val = torch.where(index < i, torch.zeros(len(x)), x)
		val[i] = 1
	return val


def log_sum_exp(x):
    """Compute log(sum(exp(x))) in a numerically stable way.
    Examples
    --------
    >>> x = [0, 1, 0]
    >>> log_sum_exp(x) #doctest: +ELLIPSIS
    1.551...
    >>> x = [1000, 1001, 1000]
    >>> log_sum_exp(x) #doctest: +ELLIPSIS
    1001.551...
    >>> x = [-1000, -999, -1000]
    >>> log_sum_exp(x) #doctest: +ELLIPSIS
    -998.448...
    """
    x = np.asarray(x)
    a = max(x)
    return a + np.log(sum(np.exp(x - a)))

def log_mean_exp(x):
    """Compute log(mean(exp(x))) in a numerically stable way.
    Examples
    --------
    >>> x = [1, 2, 3]
    >>> log_mean_exp(x) #doctest: +ELLIPSIS
    2.308...
    """
    return log_sum_exp(x) - np.log(len(x))

def log_diff_exp(x):
    """Compute log(diff(exp(x))) in a numerically stable way.
    Examples
    --------
    >>> log_diff_exp([1, 2, 3]) #doctest: +ELLIPSIS
    array([ 1.5413...,  2.5413...])
    >>> [np.log(np.exp(2)-np.exp(1)), np.log(np.exp(3)-np.exp(2))] #doctest: +ELLIPSIS
    [1.5413..., 2.5413...]
    """
    x = np.asarray(x)
    a = max(x)
    return a + np.log(np.diff(np.exp(x - a)))

def log_std_exp(x, log_mean_exp_x=None):
    """Compute log(std(exp(x))) in a numerically stable way.
    Examples
    --------
    >>> x = np.arange(8.)
    >>> print x
    [ 0.  1.  2.  3.  4.  5.  6.  7.]
    >>> log_std_exp(x) #doctest: +ELLIPSIS
    5.875416...
    >>> np.log(np.std(np.exp(x))) #doctest: +ELLIPSIS
    5.875416...
    """
    x = np.asarray(x)
    m = log_mean_exp_x
    if m is None:
        m = log_mean_exp(x)
    M = log_mean_exp(2. * x)
    return 0.5 * log_diff_exp([2. * m, M])[0]


def count_units(dataset):

    p = torch.zeros(dataset[0].size())
    N = len(dataset)

    for i in range(N):
        p += dataset[i] / N

    return p


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

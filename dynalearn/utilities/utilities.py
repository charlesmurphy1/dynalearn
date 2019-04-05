"""

utilities.py

Created by Charles Murphy on 21-08-11.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines a variety of useful functions for bm use and training.
"""

import os
import torch
import numpy as np
from math import floor, log2

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


def increment_bit(x, base=2):

    val = x * 1
    for i in range(base):
        if i in x:
            i = np.min(np.where(x==i)[0])
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
    m1 = log_mean_exp_x
    if m1 is None:
        m1 = log_mean_exp(x)
    m2 = log_mean_exp(2. * x)

    return 0.5 * log_diff_exp([2. * m1, m2])[0] 


def count_units(dataset):

    p = torch.zeros(dataset[0].size())
    N = len(dataset)

    for i in range(N):
        p += dataset[i] / N

    return p


def exp_mov_avg(x, graining):
    # cumsum = np.cumsum(np.insert(x, 0, 0)) 
    # return (cumsum[N:] - cumsum[:-N]) / float(N)
    s = np.array([x[0]]*len(x))
    for i, xx in enumerate(x[1:]):
        s[i+1] = graining*xx + (1 - graining)*s[i]
    return s

def increment_filename(path, name):
    for filename in os.listdir(path):
            if filename.startswith(name):
                is_not_finished = True
                i = 0
                while(is_not_finished):
                    f = os.path.join(path, "{0}_{1}".format(name, i))
                    if os.path.exists(f):
                        i+= 1
                    else:
                        name += "_{0}".format(i)
                        is_not_finished = False

    return name

def increment_path(path):
    is_not_finished = os.path.exists(path)
    i = 0
    while(is_not_finished):
        new_path = "{0}_{1}".format(path, i)
        if os.path.exists(new_path):
            i+= 1
        else:
            path = new_path
            is_not_finished = False

    return path


if __name__ == '__main__':
    path = "."
    name = "test"
    ext = "mcf"

    new_name = increment_filename(path, name, ext)

    print(new_name)
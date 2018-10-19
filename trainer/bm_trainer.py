"""
bm_trainer.py
Created by Charles Murphy on 01-08-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada
Defines the class BM_trainer. This class generate training procedures for BMs.
"""

from copy import copy
import torch
from history import *
from dataset import random_split, RandomSampler_with_length
from random import choice
import progressbar



class BM_trainer(object):
    """docstring for BM_trainer"""
    def __init__(self, bm, model_name="bm", history=None, weight_decay=1e-3, momentum=0.9):
        super(BM_trainer, self).__init__()
        self.bm = bm
        self.model_name = model_name

        if history is None:
            self.history = BM_History(None, None)
        else:
            self.history = history

        self.weight_decay = weight_decay
        self.momentum = momentum

        self.use_cuda = self.bm.use_cuda

        self.params_momentum = {}
        self.params_best = {}

        for k in self.bm.params:
            self.params_momentum[k] = torch.zeros(self.bm.params[k].value.size())
            self.params_best[k] = copy(self.bm.params[k])
            
        if self.use_cuda:
            for k in self.bm.params:
                self.params_momentum[k] = self.params_momentum[k].cuda()


    def compute_grad(self, v_data, num_steps):

        grad = {}
        units = self.bm.init_units({self.bm.v_key: v_data})

        # Calculating positive and negative phases
        posph = self.bm.positive_phase(v_data)
        negph = self.bm.negative_phase(None, num_steps)

        # Updating momenta
        for k in self.bm.params:
            self.params_momentum[k] *= self.momentum
            self.params_momentum[k] += posph[k] - negph[k]
            grad[k] = self.params_momentum[k]

        return grad


    def keys_to_eval(self, time, eval_step):
        x = []

        for k in self.history.statistics:
            if eval_step[k] is time:
                x.append(k)

        return x


    def save_stats(self):
        return 0


    def train(self, dataset, val_dataset=None, n_epoch=10, patience=None, batchsize=32, keep_best=True, 
              lr=None, num_steps=1, val_prop=0., eval_step=1, sample_size=100, save=False, show=False,
              path=None, verbose=False):

        if type(eval_step) is not dict:
            value = eval_step
            eval_step = {}
            for p in self.history.properties:
                eval_step[p] = value


        if lr is None:
            lr = [self.lr] * n_epoch
        else:
            if type(lr) is float:
                lr = [lr] * n_epoch
            elif type(lr) is list:
                if len(lr) != n_epoch:
                    raise ValueError("lr must be of the same size as n_epoch.")

        if val_dataset is None:
            train_dataset, val_dataset = random_split(dataset, val_prop)
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=batchsize,
                                             shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=batchsize,
                                             shuffle=True)

        train_list = list(train_loader)
        val_list = list(val_loader)
        complete_data = {"train": train_list, "val": val_list, "grad": 0}
        best = 0
        num_update = 0


        self.bm._log_Z()
        self.history.evaluate_stats(num_update, complete_data, self.bm)
        if patience is None: patience = n_epoch
        epoch_patience = 0
        for i in range(n_epoch):
            if verbose:
                widgets = [ "Epoch {} ".format(i),
                            "Update ", progressbar.Counter(), ' ',
                            progressbar.Bar('-'), ' ',
                            progressbar.Percentage(), ' ',
                            progressbar.ETA()]
                bar = progressbar.ProgressBar(widgets=widgets, maxval=len(train_loader)).start()
            grads = [0.] * len(train_loader)


            for j, v_batch in enumerate(train_loader):
                grads[j] = self.compute_grad(v_batch, num_steps)
                self.bm.update_params(grads[j], lr[i], self.weight_decay)

                sample_data = {"train": [v_batch], "val": [choice(val_list) for i in range(batchsize)]}
                self.history.estimate_stats(num_update, sample_data, self.bm)

                if verbose: bar.update(j)
                num_update += 1

            if verbose: bar.finish()



            self.history.evaluate_stats(num_update, complete_data, self.bm)
            if keep_best:
                if self.history.is_current_best():
                    for k in self.bm.params:
                        self.params_best[k] = copy(self.bm.params[k])
                    epoch_patience = 0
                else:
                    epoch_patience += 1


            if verbose: print(str(self.history))

            if epoch_patience > patience: break


        self.history.make_plots(keys=None, save=save, show=show, showbest=True)
        self.history.save_stats(path)

        if keep_best:
            if verbose: print("Best {} \n".format(self.history.best_update) + \
                              self.history.str_update(self.history.best_update))

            for k in self.bm.params:
                self.bm.params[k] = copy(self.params_best[k])
            self.model_name = "best_" + self.model_name

        self.bm.save_params(path + self.model_name + ".pt")



        return self.history, val_list


def layerwise_trainer(bm, train_dataset, n_epoch, num_steps=1, lr=None, verbose=False):
    
    return 0


if __name__ == '__main__':
    from rbm import RBM, RBM_no_Weight, RBM_no_Bias
    from dataset import BernoulliDataset, random_split
    import utilities as util
    from statistics import *


    n_visible = 100
    n_hidden = 100
    n_sample = 10000
    dataset = BernoulliDataset(n_sample, n_visible, p=None)
    init_scale = 0.01
    
    batchsize = 32
    use_cuda = False

    complete_dataset = torch.cat(dataset.data)
    p = util.count_units(dataset)
    # p = None

    with_param = True
    with_grad = True
    with_logp = True
    with_free_energy = True
    with_recon = True


    def train(bm, path, n_epoch=10, lr_min=1e-6, lr_max=1e-3):

        statstics = {}
        eval_step = {}
        if with_param:
            for k in bm.params:
                statstics[("param", k)] = Parameter_Statistics(k)
                eval_step[("param", k)] = 1

        if with_grad:
            for k in bm.params:
                statstics[("grad", k)] = Gradient_Statistics(k)
                eval_step[("grad", k)] = 1

        if with_logp:
            statstics["log-p"] = LogLikelihood_Statistics(recompute=False, graining=50)
            eval_step["log-p"] = "update"

        if with_free_energy:
            statstics["free_energy"] = Free_Energies_Statistics(graining=50)
            eval_step["free_energy"] = "update"

        if with_recon:
            statstics["recon"] = Reconstruction_MSE_Statistics(graining=50)
            eval_step["recon"] = "update"
        h = BM_History(statstics, path)
        bm_trainer = BM_trainer(bm, 1e-3,
                                history=h,
                                weight_decay=1e-4,
                                momentum=0.99)

        lr = [lr_max / (i + 1) + (i + 1) / n_epoch * lr_min for i in range(n_epoch)]

        bm_trainer.train(dataset, n_epoch, lr=lr, batchsize=32, save=True, show=False, num_steps=1, eval_step=eval_step)

    # print("RBM Training\n============")
    # path = "testdata/rbm_training/"
    # rbm = RBM(n_visible, n_hidden, batchsize, init_scale, p, use_cuda)
    # train(rbm, path)

    print("RBM no weight Training\n======================")
    path = "testdata/rbm_no_w_training/"
    rbm_no_w = RBM_no_Weight(n_visible, n_hidden, batchsize, p, use_cuda)
    train(rbm_no_w, path)

    # print("RBM no bias Training\n======================")
    # path = "testdata/rbm_no_b_training/"
    # rbm_no_w = RBM_no_Bias(n_visible, n_hidden, batchsize, init_scale, use_cuda)
    # train(rbm_no_w, path)

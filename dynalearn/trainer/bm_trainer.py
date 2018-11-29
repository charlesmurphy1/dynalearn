"""
bm_trainer.py
Created by Charles Murphy on 01-08-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada
Defines the class BM_trainer. This class generate training procedures for BMs.
"""

from copy import copy
import torch
from random import choice
import progressbar

from .history import History
from ..dynamics.dataset import random_split
from ..utilities.utilities import random_binary


class BM_trainer(object):
    """docstring for BM_trainer"""
    def __init__(self, bm, history=None, config=None):
        super(BM_trainer, self).__init__()
        self.bm = bm

        self.history = history
        if self.history is None:
            self.history = History()

        if self.history is None:
            self.history = Config()

        lr = config.LEARNING_RATE
        wd = config.WEIGHT_DECAY

        self.optimizer = config.OPTIMIZER(bm.parameters())
        self.lrs = config.LR_SCHEDULER(self.optimizer)
        self.batchsize = config.BATCHSIZE

        self.numsteps = config.NUMSTEPS
        self.numepochs = config.NUMEPOCHS
        self.keep_best = config.KEEPBEST
        self.with_pcd = config.WITH_PCD
        self.makeplot = config.MAKEPLOT
        self.verbose = config.VERBOSE

            
    def setup_dataset(self, train_dataset, val_dataset):
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=self.batchsize,
                                             shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=self.batchsize,
                                             shuffle=True)

        return train_loader, val_loader


    def setup_progbar(self, epoch, num_step):
        widgets = [ "Epoch {} ".format(epoch),
                            "Update ", progressbar.Counter(), ' ',
                            progressbar.Bar('-'), ' ',
                            progressbar.Percentage(), ' ',
                            progressbar.ETA()]
        return progressbar.ProgressBar(widgets=widgets, maxval=num_step).start()

    def evaluate_stats(self, update, data):
        self.history.evaluate_stats(update, data, self.bm)


    def estimate_stats(self, update, data):
        self.history.estimate_stats(update, data, self.bm)


    def loss(self, v_data):

        pos_units = self.bm.init_units(self.bm.inference(v_data))
        # print(self.bm.params["h"].energy_term(pos_units))
        for k in pos_units:
            if pos_units[k].u_kind == "hidden":
                new_data = random_binary(pos_units[k].data)
                pos_units[k].data = new_data
        pos_phase = self.bm.energy(pos_units)


        neg_units = self.bm.init_units({"v": v_data.clone()})
        neg_units = self.bm.sampler(neg_units, self.numsteps, self.with_pcd)
        neg_phase = self.bm.energy(neg_units)

        return pos_phase - neg_phase

    def save_stats(self):
        return 0

    def train(self, train_dataset, val_dataset):

        # Setting up dataset
        train_loader, val_loader = self.setup_dataset(train_dataset,
                                                      val_dataset)
        train_list = list(train_loader)
        val_list = list(val_loader)
        complete_data = {"train": train_list, "val": val_list, "grad": []}

        # Setting counters
        best = 0
        update = 0

        # First evaluation
        self.history.evaluate_stats(update, complete_data, self.bm)
        if self.verbose: print(f"Begin training\n{self.history}")

        # Learning phase
        for i in range(self.numepochs):
            if self.verbose: bar = self.setup_progbar(i, len(train_loader))
            grad = []

            ## Parameter updates
            for j, v_data in enumerate(train_loader):
                # print(len(v_data))
                ### Gradient ascent
                loss = self.loss(v_data)
                loss.backward()
                self.optimizer.step()
                local_grad = {}
                for k in self.bm.params:
                    local_grad[k] = self.bm.params[k].param.grad.detach().clone()
                grad.append(local_grad)
                ## Estimating model performance after update
                data = {
                        "train": [v_data],
                        "val": [choice(val_list) for i in range(self.batchsize)],
                        "grad": local_grad
                       }
                self.history.estimate_stats(update, data, self.bm)
                self.optimizer.zero_grad()

                if self.verbose: bar.update(j)
                update += 1
            complete_data["grad"] = grad
            self.bm.log_Z_to_be_eval = True

            if self.verbose: bar.finish()


            ## Evaluating model performance after epoch
            self.evaluate_stats(update, complete_data)
            criterion_val = self.history.criterion.stat["val"][update]
            self.lrs.step(criterion_val)

            ## Check for best configuration candidate
            ## Saving best configuration 
            if self.keep_best:
                if self.history.is_current_best():
                    for k in self.bm.params:
                        new_p = self.bm.params[k].param.data.clone()
                        self.history.best_params[k] = new_p
                self.bm.save_params()
            else:
                self.bm.save_params()

            ## Saving history 
            self.history.save()



            if self.verbose: print(str(self.history))

        # Get best configuration
        if self.keep_best:
            best_epoch = int(self.history.best_update / len(train_loader)) -1
            if self.verbose: print("Best epoch {} \n".format(best_epoch) + \
                              self.history.str_update(self.history.best_update))

            for k in self.bm.params:
                best_p = self.history.best_params[k].clone()
                self.bm.params[k].param.data = best_p



def layerwise_trainer(bm, train_dataset, n_epoch, num_steps=1, lr=None, verbose=False):
    
    return 0




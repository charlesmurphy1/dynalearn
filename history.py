import matplotlib.pyplot as plt
import os

class BM_History(object):
    """docstring for History"""
    def __init__(self, statistics=None, criterion=None, path=None):
        if statistics is None:
            self.statistics = {}
        else:
            self.statistics = statistics

        for k in self.statistics:
            if type(self.statistics[k]) is type(criterion):
                self.criterion = self.statistics[k]
                break

        self.best_update = -1
        self.current_update = 0
        self.path = path

        if (path is not None) and (not os.path.exists(self.path)):
            os.makedirs(self.path)


    def __str__(self):

        return self.str_update()

    def str_update(self, update=None):

        train_string = "Train. stats : "
        val_string = "Valid. stats : "
        model_string = "Model stats : "

        for p in self.statistics:
            if type(self.statistics[p].str_update(update)) is tuple:
                train_str, val_str = self.statistics[p].str_update(update)
                train_string += "\t" + train_str
                val_string += "\t" + val_str
            elif self.statistics[p].str_update(update) != "":
                    model_str = self.statistics[p].str_update(update)
                    model_string += "\t" + model_str

        result_string = ""

        if train_string != "Training stats : ":
            result_string += train_string + "\n"

        if val_string != "Validation stats : ":
            result_string += val_string + "\n"

        if model_string != "Model stats : ":
            result_string += model_string + "\n"

        return result_string

    def evaluate_stats(self, update, data, bm):
        for k in self.statistics:
            self.statistics[k].evaluate(update, data, bm)

        if self.best_update == -1: self.best_update = update

        self.current_update = update

    def estimate_stats(self, update, data, bm):
        for k in self.statistics:
            self.statistics[k].estimate(update, data, bm)

    def make_plots(self, keys=None, save=False, show=False, showbest=True):
        if keys is None:
            keys = self.statistics.keys()

        if save:
            path = self.path
        else:
            path = None
        
        best = None
        if showbest and self.best_update >= 0: best = self.best_update

        for k in keys:
            self.statistics[k].plot_stat(path, best)

        if show:
            plt.show()


    def save_stats(self, path=None):
        if path is not None: 
            for k in self.statistics:
                self.statistics[k].save_stat(path)


    def is_current_best(self):

        current = self.criterion.current_update
        best = self.best_update * 1

        current_is_better = self.criterion.is_better(current, best)

        if current_is_better:
            self.best_update = current * 1
            return True
        else:
            return False





if __name__ == '__main__':
    from rbm import RBM
    import utilities as util
    from dataset import BernoulliDataset
    from statistics import *


    n_visible = 100
    n_hidden = 100
    batchsize = 1

    bm = RBM(n_visible, n_hidden, batchsize)
    dataset = BernoulliDataset(numsample=10000, dim=n_visible, p=None)
    p = util.count_units(dataset)
    # p = None

    statistics = {"w_vh": Parameter_Statistics(("v", "h")),
                  "b_v": Parameter_Statistics("v"),
                  "b_h": Parameter_Statistics("h")}


    h = BM_History(statistics, "testdata/")

    update = 1
    for i in range(update):
        bm.init_weights(0.01)
        bm.init_biases(p)

        data = {}
        h.evaluate_stats(i, data, bm, keys=["w_vh", "b_v", "b_h"])

    h.make_plots(show=True)



                    



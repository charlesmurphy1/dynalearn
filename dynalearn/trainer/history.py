import matplotlib.pyplot as plt
import os


__all__ = ['History']

class History(object):
    """
    Base class for history monitoring.

    Monitores a set of statistics during learning

    **Parameters**
    statistics : Dict : (default = ``None``)
        Dictionary of Statistics objects and sub-classes.

    criterion : Statistics : (default = ``None``)
        Statistics used for early stopping.

    path : String : (default = ``None``)
        Path where to save all statstics. If ``None``, it does not save files.
    """
    def __init__(self, statistics=None, criterion=None, path_to_stat=None):
        """
        Initialize History object.
        """
        if statistics is None:
            self.statistics = {}
        else:
            self.statistics = statistics

        for k in self.statistics:
            if type(self.statistics[k]) is type(criterion):
                self.criterion = self.statistics[k]
                break

        self.best_update = 0
        self.best_params = {}
        self.path = path_to_stat

        if (self.path is not None) and (not os.path.exists(self.path)):
            os.makedirs(self.path)


    def __str__(self):
        """
        Displays statistics strings for latest update.
        """
        return self.str_update()

    def str_update(self, update=None):
        """
        Renders the String of all statistics.

        **Parameters**
        update : Integer : (default = ``None``)
            Statistics of specific update to be rendered in String. If ``None``,
            renders latest update.

        **Returns**
        str_of_update : String

        """
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
        """
        Evaluates all history statistics.

        **Parameters**
        update : Integer
            Update at which the statistics is evaluated.

        data : Dict
            Available data to evaluate stat.

        ..note::
            Usually generated by a ``Trainer`` object, it contains the keys
            [``"train"``, ``"val"``, ``"grad"``] corresponding to the training 
            dataset, validation dataset and the gradient of each parameter.

        model : Model
            Model object on which the statistics is performed.

        """
        for k in self.statistics:
            self.statistics[k].evaluate(update, data, bm)



    def estimate_stats(self, update, data, bm):
        """
        Estimates all history statistics.

        **Parameters**
        update : Integer
            Update at which the statistics is estiamted.

        data : Dict
            Available data to estimate stat.

        ..note::
            Usually generated by a ``Trainer`` object, it contains the keys 
            [``"train"``, ``"val"``, ``"grad"``] corresponding to the training 
            dataset, validation dataset and the gradient of each parameter.

        model : Model
            Model object on which the statistics is performed.

        """
        for k in self.statistics:
            self.statistics[k].estimate(update, data, bm)

    def make_plots(self, save=False, show=False, showbest=True):
        """
        Generate plots of all statistics

        **Parameters**
        save : Bool : (default = ``False``)
            If ``True``, it saves all plots.

        show : Bool : (default = ``False``)
            If ``True``, it reveals all plots.

        showbest : Bool : (default = ``True``)
            If ``True``, it displays best update on plots.

        """
        if save:
            path = self.path
        else:
            path = None
        
        best = None
        if showbest and self.best_update >= 0: best = self.best_update

        for k in self.statistics:
            self.statistics[k].plot_stat(path, best)

        if show:
            plt.show()


    def save_stats(self):
        """
        Saves all statistics.
        """
        if self.path is not None: 
            for k in self.statistics:
                self.statistics[k].save_stat(self.path)


    def is_current_best(self):
        """
        Determines if latest update is better than current best.

        """
        if self.criterion is not None:
            current = self.criterion.latest_update
            best = self.best_update * 1

            current_is_better = self.criterion.is_better(current, best)

            if current_is_better:
                self.best_update = current * 1
                return True
        return False





if __name__ == '__main__':
    from ..model.rbm import RBM_BernoulliBernoulli
    from ..utilities.utilities import count_units
    from ..dynamics.dataset import BernoulliDataset
    from .statistics import *


    n_visible = 100
    n_hidden = 100
    batchsize = 1

    bm = RBM_BernoulliBernoulli(n_visible, n_hidden)
    dataset = BernoulliDataset(numsample=10000, dim=n_visible, p=None)
    p = count_units(dataset)
    # p = None

    statistics = {"w_vh": Parameter_Statistics(("v", "h")),
                  "b_v": Parameter_Statistics("v"),
                  "b_h": Parameter_Statistics("h")}


    h = History(statistics, "testdata/")

    update = 1
    for i in range(update):
        bm.init_weights(0.01)
        bm.init_biases(p)

        data = {}
        h.evaluate_stats(i, data, bm, keys=["w_vh", "b_v", "b_h"])

    h.make_plots(show=True)



                    



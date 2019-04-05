import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys


class History(object):
    """
    Base class for history monitoring.

    Monitores a set of metrics during learning

    **Parameters**
    metrics : Dict : (default = ``None``)
        Dictionary of metrics objects and sub-classes.

    """
    def __init__(self, name=""):
        """
        Initialize History object.
        """

        self.name = name

        self.train_measures = {}
        self.val_measures = {}
        self.model_measures = {}


        self.current_epoch = 0

        self.train_color = '#2166ac' 
        self.val_color = '#d6604d'
        self.model_color = '#2166ac'

        # if (self.path is not None) and (not os.path.exists(self.path)):
        #     os.makedirs(self.path)

    def progression(self, epoch, time, training_metrics=None, model_metrics=None,
                    is_best=False, show_var=True):
        if is_best: 
            sys.stdout.write("New best Epoch: {0} ".format(epoch) +\
                             "- Time: {0:0.02f}".format(time))
        else: 
            sys.stdout.write("Epoch: {0} ".format(epoch) +\
                             "- Time: {0:0.02f}".format(time))

        if 'lr' in self.model_measures:
            lr = self.model_measures['lr'][-1][1]
            sys.stdout.write(", lr: {0}".format(lr))
        elif 'learning_rate' in self.model_measures:
            lr = self.model_measures['learning_rate'][-1][1]
            sys.stdout.write(", lr: {0}".format(lr))
        sys.stdout.write("\n")

        if is_best:
            if training_metrics and self.train_measures:
                sys.stdout.write("\t Train. - ")
                for m in training_metrics:
                    if m in self.train_measures:
                        mean = self.train_measures[m][-1][1]
                        std = self.train_measures[m][-1][2]
                        sys.stdout.write("{0}: {1:0.4f}".format(m, mean))
                        if show_var:
                            sys.stdout.write("{0}: {1:0.4f}".format(m, std))
                        sys.stdout.write(", ")
                sys.stdout.write("\n")

            if training_metrics and self.val_measures:
                sys.stdout.write("\t Val. - ")
                for m in training_metrics:
                    if m in self.val_measures:
                        mean = self.val_measures[m][-1][1]
                        std = self.val_measures[m][-1][2]
                        sys.stdout.write("{0}: {1:0.4f}".format(m, mean))
                        if show_var:
                            sys.stdout.write("{0}: {1:0.4f}".format(m, std))
                        sys.stdout.write(", ")
                sys.stdout.write("\n")

            if model_metrics and self.model_measures:
                sys.stdout.write("\t Model - ")
                for m in model_metrics:

                    if m in self.model_measures:
                        mean = self.model_measures[m][-1][1]
                        std = self.model_measures[m][-1][2]
                        sys.stdout.write("{0}: {1:0.4f}".format(m, mean))
                        if show_var:
                            sys.stdout.write("{0}: {1:0.4f}".format(m, std))
                        sys.stdout.write(", ")
                sys.stdout.write("\n")
        sys.stdout.flush()


    def evaluate_metrics(self, epoch, train_measures,
                         val_measures={},
                         model_measures={}):

        for m in train_measures:
            if m in self.train_measures: 
                self.train_measures[m].append([epoch,
                                               train_measures[m][0],
                                               train_measures[m][1]])
            else:
                self.train_measures[m] = [[epoch,
                                          train_measures[m][0],
                                          train_measures[m][1]]]

        for m in val_measures:
            if m in self.val_measures: 
                self.val_measures[m].append([epoch,
                                             val_measures[m][0],
                                             val_measures[m][1]])
            else:
                self.val_measures[m] = [[epoch,
                                         val_measures[m][0],
                                         val_measures[m][1]]]

        for m in model_measures:
            if m in self.model_measures: 
                self.model_measures[m].append([epoch,
                                               model_measures[m][0],
                                               model_measures[m][1]])
            else:
                self.model_measures[m] = [[epoch,
                                           model_measures[m][0],
                                           model_measures[m][1]]]

        if val_measures:
            criterion = val_measures["loss"][0]
        elif train_measures:
            criterion = train_measures["loss"][0]
        else:
            criterion = np.inf


        return criterion


    def plot(self, training_metrics=[], model_metrics=[],
              ax=None, path=None, show=False):
        """
        Generate plots of all metrics

        **Parameters**
        save : Bool : (default = ``False``)
            If ``True``, it saves all plots.

        show : Bool : (default = ``False``)
            If ``True``, it reveals all plots.

        showbest : Bool : (default = ``True``)
            If ``True``, it displays best update on plots.

        """
        if ax is None:
            fig = {}
            ax = {}
            for m in training_metrics:
                fig[m], ax[m] = plt.subplots(1, 1)
            for m in model_metrics:
                fig[m], ax[m] = plt.subplots(1, 1)
        else:
            fig, ax = ax[0], ax[1]

        for m in training_metrics:
            if m in self.train_measures:
                x_train = np.array(self.train_measures[m])
            else:
                x_train = None
            if m in self.val_measures:
                x_val = np.array(self.val_measures[m])
            else:
                x_val = None

            if x_train is not None:
                ax[m].plot(x_train[:,0], x_train[:,1], c=self.train_color, marker='o', linestyle='-', linewidth=2, markersize=5)
            if x_val is not None:
                ax[m].plot(x_val[:,0], x_val[:,1], c=self.val_color, marker='o', linestyle='-', linewidth=2, markersize=5)
            ax[m].set_ylabel(m)
            ax[m].set_xlabel('Epoch')

        for m in model_metrics:
            if m in self.model_measures:
                x = np.array(self.model_measures[m])
            else:
                x = None

            if x is not None:
                ax[m].plot(x[:,0], x[:,1], c=self.model_color, marker='o', linestyle='-', linewidth=2, markersize=5)
            ax[m].set_ylabel(m)
            ax[m].set_xlabel('Epoch')
        if show:
            plt.show()

        if path:
            if type(fig) == dict:
                for m in training_metrics:
                    fig[m].savefig(os.path.join(path, 'png', m + '.png'))
                    fig[m].savefig(os.path.join(path, 'pdf', m + '.pdf'))

                for m in model_metrics:
                    fig[m].savefig(os.path.join(path, 'png', m + '.png'))
                    fig[m].savefig(os.path.join(path, 'pdf', m + '.pdf'))
            else:
                fig.savefig(os.path.join(path, 'png', self.name + '.png'))
                fig.savefig(os.path.join(path, 'pdf', self.name + '.pdf'))


        return fig, ax


    def save(self, path):
        """
        Saves metrics.
        """
        with open(path, "wb") as f:
            data = {}
            data['train'] = self.train_measures
            data['val'] = self.val_measures
            data['model'] = self.model_measures
            data['current-epoch'] = self.current_epoch
            pickle.dump(data, f)


    def load(self, path):
        """
        Load metrics.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.train_measures = data['train']
            self.val_measures = data['val']
            self.model_measures = data['model']
            self.current_epoch = data['current-epoch']
        
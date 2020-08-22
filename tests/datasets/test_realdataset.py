import networkx as nx
import numpy as np
import torch

from dynalearn.config import ExperimentConfig
from dynalearn.experiments import Experiment
from unittest import TestCase


class RealsDatasetTest(TestCase):
    def setUp(self):
        self.config = ExperimentConfig.test(config="discrete")
        self.num_networks = 2
        self.num_samples = 10
        self.num_nodes = 10
        self.batch_size = 5
        self.config.train_details.num_networks = self.num_networks
        self.config.train_details.num_samples = self.num_samples
        self.config.networks.num_nodes = self.num_nodes
        self.config.networks.p = 0.5
        self.exp = Experiment(self.config, verbose=0)
        self.dataset = self.exp.dataset
        self.dataset.setup(self.exp)
        data = self.dataset._generate_data_(self.exp.train_details)
        self.dataset.data = data

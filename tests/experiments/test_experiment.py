import dynalearn as dl
import numpy as np
import unittest


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.config = dl.ExperimentConfig.test()
        self.experiment = dl.Experiment(self.config, verbose=0)

    def test_generate(self):
        self.experiment.generate_data()
        expected = (
            self.config.train_details.num_samples
            * self.config.train_details.num_networks
        )
        actual = len(self.experiment.dataset)
        self.assertEqual(expected, actual)

    def test_train(self):
        self.experiment.generate_data()
        self.experiment.train_model()
        logs = self.experiment.model.nn.history._epoch_logs
        for epoch, log in logs.items():
            self.assertFalse(np.isnan(log["loss"]))

    def test_compute_metrics(self):
        self.experiment.generate_data()
        self.experiment.compute_metrics()

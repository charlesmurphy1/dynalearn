import dynalearn as dl
import getpass
import os
import numpy as np
import tensorflow as tf


path_to_dir = (
    "/home/"
    + getpass.getuser()
    + "/Documents/ulaval/doctorat/projects/dynalearn/data/training/"
)
path_to_models = (
    "/home/"
    + getpass.getuser()
    + "/Documents/ulaval/doctorat/projects/dynalearn/data/models/"
)
if os.path.exists("/media/" + getpass.getuser() + "/LaCie/"):
    path_to_dir = "/media/" + getpass.getuser() + "/LaCie/dynalearn-data/training/"
    path_to_models = "/media/" + getpass.getuser() + "/LaCie/dynalearn-data/models/"


dynamics_models = [
    {"name": "SIS", "params": {"infection": 0.04, "recovery": 0.08, "init": "None"}},
    {
        "name": "PlanckSIS",
        "params": {"temperature": 6, "recovery": 0.08, "init": "None"},
    },
    {
        "name": "SISSIS",
        "params": {
            "infection1": 0.04,
            "infection2": 0.03,
            "recovery1": 0.08,
            "recovery2": 0.1,
            "coupling": 5,
            "init": "None",
        },
    },
]
model_configs = [
    dl.models.GNNConfig.SISGNN(),
    dl.models.GNNConfig.SISGNN(),
    dl.models.GNNConfig.SISSISGNN(),
]
metric_configs = [
    dl.metrics.MetricsConfig.SISMetrics(),
    dl.metrics.MetricsConfig.PlanckSISMetrics(),
    dl.metrics.MetricsConfig.SISSISMetrics(),
]
graph_models = [
    {"name": "ERGraph", "params": {"N": 1000, "density": 0.004}},
    {"name": "BAGraph", "params": {"N": 1000, "M": 2}},
]

num_samples = [100]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

for dynamics, model, metric in zip(dynamics_models, model_configs, metric_configs):
    for graph in graph_models:
        for n in num_samples:
            name = "{0}-{1}-{2}".format(dynamics["name"], graph["name"], n)
            print(f"Experiment - {name}")
            config = {
                "name": name,
                "graph": graph,
                "dynamics": dynamics,
                "model": {"name": "EpidemicPredictor", "config": model},
                "generator": {
                    "name": "DynamicsGenerator",
                    "config": dl.datasets.GeneratorConfig.default(),
                    "sampler": {
                        "name": "StateBiasedSampler",
                        "config": dl.datasets.samplers.SamplerConfig.BiasedSamplerDefault(
                            dynamics
                        ),
                    },
                },
                "metrics": {
                    "name": [
                        "AttentionMetrics",
                        "TrueLTPMetrics",
                        "GNNLTPMetrics",
                        "MLELTPMetrics",
                        "TrueStarLTPMetrics",
                        "GNNStarLTPMetrics",
                        "UniformStarLTPMetrics",
                        "StatisticsMetrics",
                        "PoissonEpidemicsMFMetrics",
                        "PoissonEpidemicsSSMetrics",
                    ],
                    "config": metric,
                },
                "training": dl.TrainingConfig.test(),
                "path_to_dir": path_to_dir,
                "path_to_bestmodel": path_to_models,
            }

            experiment = dl.Experiment(config)
            experiment.run()

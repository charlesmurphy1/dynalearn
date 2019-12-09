import dynalearn as dl
import getpass
import os
import numpy as np


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
if os.path.exists("media/" + getpass.getuser() + "LaCie/"):
    path_to_dir = "media/" + getpass.getuser() + "LaCie/dynalearn-data/training/"
    path_to_models = "media/" + getpass.getuser() + "LaCie/dynalearn-data/models/"


dynamics_models = [
    {"name": "SIS", "params": {"infection": 0.04, "recovery": 0.08, "init": "None"}},
    {
        "name": "SISSIS",
        "params": {
            "infection1": 0.04,
            "infectio2": 0.03,
            "recovery1": 0.08,
            "recovery2": 0.1,
            "coupling": 5,
            "init": "None",
        },
    },
]
model_configs = [dl.models.GNNConfig.SISGNN(), dl.models.GNNConfig.SISSISGNN()]
graph_models = [
    {"name": "ERGraph", "params": {"N": 1000, "p": 0.004}},
    {"name": "BAGraph", "params": {"N": 1000, "M": 2}},
]

num_samples = [100, 1000, 10000, 50000]

for dynamics, model in zip(dynamics_models, model_configs):
    for graph in graph_models:
        for n in num_samples:
            config = {
                "name": "{0}-{1}-{2}".format(dynamics["name"], graph["name"], n),
                "graph": graph,
                "dynamics": dynamics,
                "model": {"name": "EpidemicPredictor", "config": m},
                "generator": {
                    "name": "DynamicsGenerator",
                    "config": dl.datasets.GeneratorConfig.default(),
                    "sampler": {
                        "name": "StateBiasedSampler",
                        "config": dl.datasets.samplers.SamplerConfig.BiasedSamplerDefault(),
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
                    "config": dl.metrics.MetricsConfig.default(),
                },
                "training": dl.TrainingConfig.changing_num_samples(n),
                "path_to_dir": path_to_dir,
                "path_to_bestmodel": path_to_models,
            }

            experiment = dl.Experiment(config)
            experiment.run()

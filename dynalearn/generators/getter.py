from dynalearn.generators import *
from dynalearn.generators.samplers import get_sampler


def get(params_dict, graph_model, dynamics_model):
    name = params_dict["name"]
    params = params_dict["params"]
    sampler = get_sampler(dynamics_model, params_dict["sampler"])

    if name == "DynamicsGenerator":
        return DynamicsGenerator(
            graph_model,
            dynamics_model,
            sampler,
            batch_size=params["batch_size"],
            with_truth=params["with_truth"],
        )
    else:
        raise ValueError("Wrong name of generator.")

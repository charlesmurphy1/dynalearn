import dynalearn as dl

generators = ["DynamicsGenerator"]


def get(params_dict, graph_model, dynamics_model):
    name = params_dict["name"]
    params = params_dict["params"]
    sampler = dl.generators.samplers.get(params_dict["sampler"], dynamics_model)

    if name == "DynamicsGenerator":
        return dl.generators.DynamicsGenerator(
            graph_model,
            dynamics_model,
            sampler,
            batch_size=params["batch_size"],
            with_truth=params["with_truth"],
        )
    else:
        raise ValueError(
            "Wrong name of generator. Valid entries are: {0}".format(generators)
        )

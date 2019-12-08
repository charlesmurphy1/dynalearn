import dynalearn as dl

generators = ["DynamicsGenerator"]


def get(params_dict, graph_model, dynamics_model):
    name = params_dict["name"]
    config = params_dict["config"]
    sampler = dl.datasets.samplers.get(params_dict["sampler"], dynamics_model)

    if name == "DynamicsGenerator":
        return dl.datasets.DynamicsGenerator(
            graph_model, dynamics_model, sampler, config
        )
    else:
        raise ValueError(
            "Wrong name of generator. Valid entries are: {0}".format(generators)
        )

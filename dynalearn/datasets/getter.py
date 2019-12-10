import dynalearn as dl

generators = ["DynamicsGenerator"]


def get(config, graph_model, dynamics_model):
    name = config["name"]
    _config = config["config"]
    # print(config)
    sampler = dl.datasets.samplers.get(config["sampler"])
    if name == "DynamicsGenerator":
        return dl.datasets.DynamicsGenerator(
            graph_model, dynamics_model, sampler, _config
        )
    else:
        raise ValueError(
            "Wrong name of generator. Valid entries are: {0}".format(generators)
        )

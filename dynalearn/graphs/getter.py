import dynalearn as dl


graph_models = {
    "CycleGraph": dl.graphs.CycleGraph,
    "CompleteGraph": dl.graphs.CompleteGraph,
    "StarGraph": dl.graphs.StarGraph,
    "EmptyGraph": dl.graphs.EmptyGraph,
    "RegularGraph": dl.graphs.RegularGraph,
    "BAGraph": dl.graphs.BAGraph,
    "ERGraph": dl.graphs.ERGraph,
}


def get(params_dict):
    name = params_dict["name"]
    params = params_dict["params"]

    if name in graph_models:
        return dynamics_models[name](params)
    else:
        raise ValueError(
            "Wrong name of graph. Valid entries are: {0}".format(
                list(graph_models.keys())
            )
        )

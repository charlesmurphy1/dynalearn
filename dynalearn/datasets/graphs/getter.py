from .graph_generator import *


graph_models = {
    "CycleGraph": CycleGraph,
    "CompleteGraph": CompleteGraph,
    "StarGraph": StarGraph,
    "EmptyGraph": EmptyGraph,
    "RegularGraph": RegularGraph,
    "BAGraph": BAGraph,
    "ERGraph": ERGraph,
    "DegreeSequenceGraph": DegreeSequenceGraph,
}


def get(params_dict):
    name = params_dict["name"]
    params = params_dict["params"]

    if name in graph_models:
        return graph_models[name](params)
    else:
        raise ValueError(
            "Wrong name of graph. Valid entries are: {0}".format(
                list(graph_models.keys())
            )
        )

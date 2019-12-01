from dynalearn.graphs import *


def get(params_dict):
    name = params_dict["name"]
    params = params_dict["params"]
    if name == "CycleGraph":
        return CycleGraph(params["N"])
    elif name == "CompleteGraph":
        return CompleteGraph(params["N"])
    elif name == "StarGraph":
        return StarGraph(params["N"])
    elif name == "EmptyGraph":
        return EmptyGraph(params["N"])
    elif name == "RegularGraph":
        return RegularGraph(params["N"], params["degree"])
    elif name == "BAGraph":
        return BAGraph(params["N"], params["M"])
    elif name == "ERGraph":
        return ERGraph(params["N"], params["p"])
    else:
        raise ValueError("Wrong name of graph.")

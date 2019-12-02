import dynalearn as dl


def get(params_dict):
    name = params_dict["name"]
    params = params_dict["params"]
    if name == "CycleGraph":
        return dl.graphs.CycleGraph(params["N"])
    elif name == "CompleteGraph":
        return dl.graphs.CompleteGraph(params["N"])
    elif name == "StarGraph":
        return dl.graphs.StarGraph(params["N"])
    elif name == "EmptyGraph":
        return dl.graphs.EmptyGraph(params["N"])
    elif name == "RegularGraph":
        return dl.graphs.RegularGraph(params["N"], params["degree"])
    elif name == "BAGraph":
        return dl.graphs.BAGraph(params["N"], params["M"])
    elif name == "ERGraph":
        return dl.graphs.ERGraph(params["N"], params["p"])
    else:
        raise ValueError("Wrong name of graph.")

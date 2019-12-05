import dynalearn as dl


def get(param_dict, graph, dynamics):
    name = param_dict["name"]
    params = param_dict["params"]
    if name == "LocalStatePredictor":
        return dl.models.LocalStatePredictor(
            graph.num_nodes, len(dynamics.state_label), params, seed=params["tf_seed"]
        )
    else:
        raise ValueError("Wrong name of model.")

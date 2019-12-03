import dynalearn as dl


def get(param_dict, graph, dynamics):
    name = param_dict["name"]
    params = param_dict["params"]
    if name == "LocalStatePredictor":
        return dl.models.LocalStatePredictor(
            graph.num_nodes,
            len(dynamics.state_label),
            params["in_features"],
            params["attn_features"],
            params["out_features"],
            params["n_heads"],
            in_activation=params["in_activation"],
            attn_activation=params["attn_activation"],
            out_activation=params["out_activation"],
            weight_decay=params["weight_decay"],
            seed=params["tf_seed"],
        )
    else:
        raise ValueError("Wrong name of model.")

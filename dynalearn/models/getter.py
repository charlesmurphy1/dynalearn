import dynalearn as dl


def get(param_dict):
    name = param_dict["name"]
    config = param_dict["config"]
    if name == "EpidemicPredictor":
        return dl.models.EpidemicPredictor(config)
    else:
        raise ValueError("Wrong name of model.")

import dynalearn as dl


meanfields = {
    "SIS": dl.meanfields.SIS_MF,
    "SIR": dl.meanfields.SIR_MF,
    "SoftThresholdSIS": dl.meanfields.SoftThresholdSIS_MF,
    "SoftThresholdSIR": dl.meanfields.SoftThresholdSIR_MF,
    "NonLinearSIS": dl.meanfields.NonLinearSIS_MF,
    "NonLinearSIR": dl.meanfields.NonLinearSIR_MF,
    "SineSIS": dl.meanfields.SineSIS_MF,
    "SineSIR": dl.meanfields.SineSIR_MF,
    "PlanckSIS": dl.meanfields.PlanckSIS_MF,
    "PlanckSIR": dl.meanfields.PlanckSIR_MF,
    "SISSIS": dl.meanfields.SISSIS_MF,
}


def get(params, dynamics):
    name = type(dynamics).__name__

    if name in meanfields:
        return meanfields[name](params)
    else:
        raise ValueError(
            "Wrong name of meanfields. Valid entries are: {0}".format(
                list(meanfields.keys())
            )
        )

import numpy as np


def SIS_aggregator(
    in_state, summaries, mean, var=None, out_state=None, operation="mean"
):
    x = np.unique(np.sort(summaries[:, 2]))
    y = np.zeros(x.shape)
    err = np.zeros(x.shape)

    if operation == "mean":
        operation_on_mean = np.nanmean
        operation_on_var = lambda var: np.nanmean(var) / var[~np.isnan(var)].size
    elif operation == "sum":
        operation_on_mean = np.nansum
        operation_on_var = np.nansum

    for i, xx in enumerate(x):
        index = (summaries[:, 2] == xx) * (summaries[:, 0] == in_state)
        if out_state is None:
            y[i] = operation_on_mean(mean[index])
        else:
            y[i] = operation_on_mean(mean[index, out_state])

        if var is not None:
            if out_state is None:
                err[i] = np.sqrt(operation_on_var(var[index]))
            else:
                err[i] = np.sqrt(operation_on_var(var[index, out_state]))

    return x, y, err


def SIR_aggregator(
    in_state, summaries, mean, var=None, out_state=None, operation="mean"
):
    return SIS_aggregator(in_state, summaries, mean, var, out_state, operation)


def SoftThresholdSIS_aggregator(
    in_state, summaries, mean, var=None, out_state=None, operation="mean"
):
    k = np.sum(summaries[:, 1:], axis=1)
    l = summaries[:, 2]

    x = np.sort(np.unique(l / k))
    y = np.zeros(x.shape)
    err = np.zeros(x.shape)

    if operation == "mean":
        operation_on_mean = np.nanmean
        operation_on_var = lambda var: np.nanmean(var) / var[~np.isnan(var)].size
    elif operation == "sum":
        operation_on_mean = np.nansum
        operation_on_var = np.nansum

    for i, xx in enumerate(x):
        index = (l / k == xx) * (summaries[:, 0] == in_state)

        if out_state is None:
            y[i] = operation_on_mean(mean[index])
        else:
            y[i] = operation_on_mean(mean[index, out_state])

        if var is not None:
            if out_state is None:
                err[i] = np.sqrt(operation_on_var(var[index]))
            else:
                err[i] = np.sqrt(operation_on_var(var[index, out_state]))
    return x, y, err


def SoftThresholdSIR_aggregator(
    in_state, summaries, mean, var=None, out_state=None, operation="mean"
):
    return SoftThresholdSIS_aggregator(
        in_state, summaries, mean, var, out_state, operation
    )


def SIS_aggregator(
    in_state, summaries, mean, var=None, out_state=None, operation="mean"
):
    x = np.unique(np.sort(summaries[:, 2] + summaries[:, 4]))
    y = np.zeros(x.shape)
    err = np.zeros(x.shape)

    if operation == "mean":
        operation_on_mean = np.nanmean
        operation_on_var = lambda var: np.nanmean(var) / var[~np.isnan(var)].size
    elif operation == "sum":
        operation_on_mean = np.nansum
        operation_on_var = np.nansum

    for i, xx in enumerate(x):
        index = (summaries[:, 2] == xx) * (summaries[:, 0] == in_state)
        if out_state is None:
            y[i] = operation_on_mean(mean[index])
        else:
            y[i] = operation_on_mean(mean[index, out_state])

        if var is not None:
            if out_state is None:
                err[i] = np.sqrt(operation_on_var(var[index]))
            else:
                err[i] = np.sqrt(operation_on_var(var[index, out_state]))

    return x, y, err

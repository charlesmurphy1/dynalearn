import dynalearn as dl

param_dict = {
    "name": "test_experiment",
    "graph": {"name": "BAGraph", "params": {"N": 1000, "M": 2}},
    "dynamics": {
        "name": "SIS",
        "params": {"infection": 0.04, "recovery": 0.08, "init": "None"},
    },
    "model": {
        "name": "LocalStatePredictor",
        "params": {
            "in_features": [32],
            "attn_features": [32],
            "out_features": [32],
            "n_heads": [1],
            "in_activation": "relu",
            "attn_activation": "relu",
            "out_activation": "relu",
            "weight_decay": 0.0001,
            "tf_seed": 2,
        },
    },
    "generator": {
        "name": "DynamicsGenerator",
        "params": {
            "batch_size": -1,
            "num_graphs": 1,
            "num_sample": 100,
            "resampling_time": 2,
            "val_fraction": 0.01,
            "max_null_iter": 1,
            "with_truth": 0,
        },
        "sampler": {
            "name": "StateBiasedSampler",
            "params": {
                "sampling_bias": 0.6,
                "val_bias": 0.8,
                "replace": 1,
                "resample": 1000,
                "sample_from_weights": 0,
            },
        },
    },
    "config": {
        "optimizer": "Adam",
        "initial_lr": 0.0005,
        "loss": "categorical_crossentropy",
        "schedule": {"epoch": 10, "factor": 2},
        "epochs": 1,
        "np_seed": 1,
    },
    "metrics": [
        # "AttentionMetrics",
        # "TrueLTPMetrics",
        # "GNNLTPMetrics",
        # "MLELTPMetrics",
        # "TrueStarLTPMetrics",
        # "GNNStarLTPMetrics",
        # "StatisticsMetrics",
        "PoissonEpidemicsMFMetrics",
        "PoissonEpidemicsSSMetrics",
    ],
    "path_to_dir": "./data/ba-sis/",
    "filename_data": "data.h5",
    "filename_metrics": "metrics.h5",
    "filename_model": "model.h5",
    "filename_bestmodel": "bestmodel.h5",
}

experiment = dl.Experiment(param_dict, verbose=1)
experiment.compute_metrics()

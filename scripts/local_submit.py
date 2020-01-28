import dynalearn as dl
import os
import time

num_samples = 1000

path_to_all = "../data"
path_to_dir = os.path.join(path_to_all, "training")
path_to_models = os.path.join(path_to_all, "models")

configs_to_run = [
    # dl.ExperimentConfig.sis_er(num_samples, path_to_dir, path_to_models),
    dl.ExperimentConfig.sis_ba(num_samples, path_to_dir, path_to_models),
    #    dl.ExperimentConfig.plancksis_er(num_samples, path_to_dir, path_to_models),
    #    dl.ExperimentConfig.plancksis_ba(num_samples, path_to_dir, path_to_models),
    # dl.ExperimentConfig.sissis_er(num_samples, path_to_dir, path_to_models),
    # dl.ExperimentConfig.sissis_ba(num_samples, path_to_dir, path_to_models),
]

for config in configs_to_run:
    path_to_data = os.path.join(path_to_dir, config.config["name"])
    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
    config.config["training"].step_per_epoch = num_samples
    config.config["metrics"]["name"] = [
        "AttentionMetrics",
        "TrueLTPMetrics",
        "GNNLTPMetrics",
        "MLELTPMetrics",
        "TrueStarLTPMetrics",
        "GNNStarLTPMetrics",
        "UniformStarLTPMetrics",
        "StatisticsMetrics",
    ]
    config.save(path_to_data)
    script = "#!/bin/bash\n"
    # script += "python training_script.py --config_path {0} --verbose {1}\n".format(
    #     config.path_to_config, 1
    # )
    script += "python summarize.py --config_path {0}\n".format(config.path_to_config)

    # seed = int(time.time())
    seed = 0
    path_to_script = "{0}/{1}-{2}.sh".format(
        "./launch_scripts", config.config["name"], seed
    )

    with open(path_to_script, "w") as f:
        f.write(script)

    os.system("bash {0}".format(path_to_script))
    os.remove(path_to_script)

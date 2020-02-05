import dynalearn as dl
import os
import time

num_samples = 100

path_to_all = "/home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/"
# path_to_all = "../data/"
path_to_dir = os.path.join(path_to_all, "test")
path_to_models = os.path.join(path_to_all, "test")

configs_to_run = [
    dl.ExperimentConfig.sis_ba(num_samples, path_to_dir, path_to_models),
]

for config in configs_to_run:
    path_to_data = os.path.join(path_to_dir, config.config["name"])
    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
    config.config["training"].step_per_epoch = num_samples
    config.config["training"].num_epochs = 1
    config.config["metrics"]["name"] = [
        "AttentionMetrics",
        "TrueLTPMetrics",
        "GNNLTPMetrics",
        "MLELTPMetrics",
        "TrueStarLTPMetrics",
        "GNNStarLTPMetrics",
        "UniformStarLTPMetrics",
        "StatisticsMetrics",
        # "TruePEMFMetrics",
        # "GNNPEMFMetrics",
        # "TruePESSMetrics",
        # "GNNPESSMetrics",
    ]
    config.config["metrics"]["config"].num_samples = 1
    config.config["metrics"]["config"].initial_burn = 1

    config.save(path_to_data)
    script = "#!/bin/bash\n"
    script += "#SBATCH --account=def-aallard\n"
    script += "#SBATCH --time=02:00:00\n"
    script += "#SBATCH --job-name=test-sis\n".format(config.config["name"])
    script += "#SBATCH --output={0}.out\n".format(os.path.join(path_to_data, "output"))
    script += "#SBATCH --gres=gpu:1\n"
    script += "#SBATCH --mem=12G\n"
    script += "\n"
    script += "module load python/3.6 scipy-stack mpi4py\n"
    script += "source ~/.dynalearn-env/bin/activate\n"
    script += "python training_script.py --config_path {0} --verbose {1}\n".format(
        config.path_to_config, 1
    )
    script += "python summarie.py --config_path {0}\n".format(config.path_to_config)
    script += "deactivate\n"

    # seed = int(time.time())
    seed = 0
    path = "{0}/{1}-{2}.bash".format("./launch_scripts", config.config["name"], seed)

    with open(path, "w") as f:
        f.write(script)

    os.system("bash {0}".format(path))
    # os.system("sbatch {0}".format(path))

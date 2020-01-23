import dynalearn as dl
import os
import time

num_samples = 10000

path_to_data = "/home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/"
path_to_dir = os.path.join(path_to_data, "training")
path_to_models = os.path.join(path_to_data, "models")

configs_to_run = [
    dl.ExperimentConfig.sis_er(num_samples, path_to_dir, path_to_models),
    dl.ExperimentConfig.sis_ba(num_samples, path_to_dir, path_to_models),
    dl.ExperimentConfig.plancksis_er(num_samples, path_to_dir, path_to_models),
    dl.ExperimentConfig.plancksis_ba(num_samples, path_to_dir, path_to_models),
    dl.ExperimentConfig.sissis_er(num_samples, path_to_dir, path_to_models),
    dl.ExperimentConfig.sissis_ba(num_samples, path_to_dir, path_to_models),
]

for config in configs_to_run:
    config.save(path_to_dir)
    script = "#!/bin/bash\n"
    script += "#SBATCH --account=def-aallard\n"
    script += "#SBATCH --time=24:00:00\n"
    script += "#SBATCH --job-name=training_{0}\n".format(config.config["name"])
    script += "#SBATCH --output={0}.out\n".format(config.config["name"])
    script += "#SBATCH --gres=gpu:1\n"
    script += "#SBATCH --mem=12G\n"
    script += "\n"
    script += "module load python/3.7 scipy-stack mpi4py\n"
    script += "source ~/pyenv/.dynalearn-env/bin/activate\n"
    script += "python training_script.py --config_path {0} --verbose {1}\n".format(
        config.path_to_config, 2
    )
    script += "deactivate\n"

    # seed = int(time.time())
    seed = 0
    path = "{0}/{1}-{2}.bash".format(
        "./launching_scripts/", config.config["name"], seed
    )

    with open(path, "w") as f:
        f.write(script)

    os.system("bash {0}".format(path))

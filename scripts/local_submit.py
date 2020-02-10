import os
import time

num_samples = 20000

path_to_all = "../data/"
path_to_dir = os.path.join(path_to_all, "training")
path_to_model = os.path.join(path_to_all, "models")
path_to_summary = os.path.join(path_to_dir, "summary")

if not os.path.exists(path_to_dir):
    os.makedirs(path_to_dir)
if not os.path.exists(path_to_model):
    os.makedirs(path_to_model)
if not os.path.exists(path_to_summary):
    os.makedirs(path_to_summary)

configs_to_run = [
    # "sis_er",
    # "sis_ba",
    # "plancksis_er",
    "plancksis_ba",
    # "sissis_er",
    # "sissis_ba",
]

for config in configs_to_run:
    name = config + "_" + str(num_samples)
    path_to_data = os.path.join(path_to_dir, name)
    script = "#!/bin/bash\n"
    script += "#SBATCH --account=def-aallard\n"
    script += "#SBATCH --time=48:00:00\n"
    script += "#SBATCH --job-name={0}\n".format(name)
    script += "#SBATCH --output={0}.out\n".format(os.path.join(path_to_data, "output"))
    script += "#SBATCH --gres=gpu:1\n"
    script += "#SBATCH --mem=24G\n"
    script += "\n"
    script += "module load python/3.6 scipy-stack mpi4py\n"
    script += "source ~/.dynalearn-env/bin/activate\n"
    script += "python training_script.py"
    script += " --config {0}".format(config)
    script += " --num_samples {0}".format(num_samples)
    script += " --path_to_data {0}".format(path_to_data)
    script += " --path_to_model {0}".format(path_to_model)
    script += " --path_to_summary {0}".format(path_to_summary)
    script += " --test 0"
    script += " --verbose 1\n"
    script += "deactivate\n"

    seed = 0
    path = "{0}/{1}-{2}.sh".format("./launch_scripts", config, seed)

    with open(path, "w") as f:
        f.write(script)

    os.system("bash {0}".format(path))

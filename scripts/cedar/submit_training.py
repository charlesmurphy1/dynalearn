import os
import time


path_to_all = "/home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/"
path_to_dir = os.path.join(path_to_all, "training")
path_to_model = os.path.join(path_to_dir, "models")
path_to_summary = os.path.join(path_to_dir, "summary")
path_to_scripts = "~/source/dynalearn/scripts/"

if not os.path.exists(path_to_dir):
    os.makedirs(path_to_dir)
if not os.path.exists(path_to_model):
    os.makedirs(path_to_model)
if not os.path.exists(path_to_summary):
    os.makedirs(path_to_summary)

num_samples = [100, 500, 1000, 5000, 10000, 20000]
configs_to_run = [
    "sis_er",
    "sis_ba",
    "plancksis_er",
    "plancksis_ba",
    "sissis_er",
    "sissis_ba",
]

for ns in num_samples:
    for config in configs_to_run:
        name = config + "_ns" + str(ns)
        path_to_data = os.path.join(path_to_dir, name)
        if not os.path.exists(path_to_data):
            os.makedirs(path_to_data)
        script = "#!/bin/bash\n"
        script += "#SBATCH --account=def-aallard\n"
        script += "#SBATCH --time=24:00:00\n"
        script += "#SBATCH --job-name={0}\n".format(name)
        script += "#SBATCH --output={0}.out\n".format(
            os.path.join(path_to_data, "output")
        )
        script += "#SBATCH --gres=gpu:p100:1\n"
        script += "#SBATCH --mem=24G\n"
        script += "\n"
        script += "module load python/3.6 scipy-stack mpi4py\n"
        script += "source ~/.dynalearn-env/bin/activate\n"
        script += "python {0}training_script.py".format(path_to_scripts)
        # script += "python ss_script.py"
        script += " --config {0}".format(config)
        script += " --num_samples {0}".format(ns)
        script += " --num_nodes {0}".format(1000)
        script += " --resampling_time {0}".format(2)
        script += " --suffix {0}".format(f"ns{ns}")
        script += " --path_to_data {0}".format(path_to_data)
        script += " --path_to_model {0}".format(path_to_model)
        script += " --path_to_summary {0}".format(path_to_summary)
        script += " --test 0"
        script += " --verbose 2\n"
        script += "deactivate\n"

        seed = 0
        path = "{0}/{1}-{2}.sh".format(
            "~/project/murphy9/dynlearn-launch", config, seed,
        )

        with open(path, "w") as f:
            f.write(script)

        os.system("sbatch {0}".format(path))

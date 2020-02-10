import os
import time

num_samples = 20000

path_to_all = "/home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/"
path_to_dir = os.path.join(path_to_all, "test")
path_to_model = os.path.join(path_to_all, "test")
path_to_summary = os.path.join(path_to_dir, "summary")

if not os.path.exists(path_to_dir):
    os.makedirs(path_to_dir)
if not os.path.exists(path_to_model):
    os.makedirs(path_to_model)
if not os.path.exists(path_to_summary):
    os.makedirs(path_to_summary)

config = "sis_er"

name = config + "_test"
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
script += " --test 1"
script += " --verbose 2\n"
script += "deactivate\n"

seed = 0
path = "{0}/{1}-{2}.sh".format(
    "/home/murphy9/source/dynalearn/scripts/launch_scripts", config, seed
)

with open(path, "w") as f:
    f.write(script)

os.system("sbatch {0}".format(path))

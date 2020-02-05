import os
import time

num_samples = 100

path_to_all = "/home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/"
# path_to_all = "../data/"
path_to_dir = os.path.join(path_to_all, "test-training")
path_to_models = os.path.join(path_to_all, "test-models")

if not os.path.exists(path_to_dir):
    os.makedirs(path_to_dir)
if not os.path.exists(path_to_models):
    os.makedirs(path_to_models)

config = "sis_er"

name = config + "_" + str(num_samples)
path_to_data = os.path.join(path_to_dir, config)
script = "#!/bin/bash\n"
script += "#SBATCH --account=def-aallard\n"
script += "#SBATCH --time=1:00:00\n"
script += "#SBATCH --job-name={0}\n".format(name)
script += "#SBATCH --output={0}.out\n".format(os.path.join(path_to_data, "output"))
script += "#SBATCH --gres=gpu:1\n"
script += "#SBATCH --mem=12G\n"
script += "\n"
script += "module load python/3.6 scipy-stack mpi4py\n"
script += "source ~/.dynalearn-env/bin/activate\n"
script += "python training_script.py"
script += " --name {0}".format(config)
script += " --num_samples {0}".format(num_samples)
script += " --path_to_data {0}".format(path_to_data)
script += " --path_to_models {0}".format(path_to_models)
script += " --verbose 1"
script += " --test 1"
script += "deactivate\n"

seed = 0
path = "{0}/{1}-{2}.sh".format("./launch_scripts", config, seed)

with open(path, "w") as f:
    f.write(script)

os.system("sbatch {0}".format(path))

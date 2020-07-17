import os
import time
from itertools import product


path_to_dynalearn = (
    "/home/charles/Documents/ulaval/doctorat/projects/dynalearn-all/dynalearn/"
)
path_to_dynalearn_data = "/home/charles/Documents/ulaval/doctorat/projects/dynalearn-all/dynalearn/data/phase2-data"
path_to_all = os.path.join(path_to_dynalearn_data, "training")
path_to_data = os.path.join(path_to_all, "full_data")
path_to_best = os.path.join(path_to_all, "best")
path_to_summary = os.path.join(path_to_all, "summary")
path_to_outputs = os.path.join(path_to_all, "outputs")

if not os.path.exists(path_to_data):
    os.makedirs(path_to_data)
if not os.path.exists(path_to_best):
    os.makedirs(path_to_best)
if not os.path.exists(path_to_summary):
    os.makedirs(path_to_summary)
if not os.path.exists(path_to_outputs):
    os.makedirs(path_to_outputs)

num_nodes = 1000
# num_samples_array = [100, 500, 1000, 5000, 10000, 20000]
num_samples_array = [1000]
wsize_array = [1]
wstep_array = [1]
# hide_prob_array = [0.0, 0.25, 0.5, 0.75 1.0]
hide_prob_array = [0.0]
config_array = [
    # "sissis-ba",
    "partiallyhiddensissis-ba",
]
tasks = ["generate_data", "train_model", "compute_metrics"]
# tasks = ["load", "generate_data", "compute_metrics"]

# metrics = ["ltp", "star-ltp", "meanfield", "stationary", "stats"]
metrics = ["ltp", "stationary", "stats"]

to_zip = [
    "config.pickle",
    # "data.h5",
    "metrics.h5",
    "history.pickle",
    "model.pt",
    "optim.pt",
]

for num_samples, config, wsize, wstep, hp in product(
    num_samples_array, config_array, wsize_array, wstep_array, hide_prob_array
):
    suffix = "ns" + str(num_samples)
    name = config + "-" + suffix + "-ws" + str(wsize) + "-wt" + str(wstep)
    script = "#!/bin/bash\n"
    # script += "#SBATCH --account=def-aallard\n"
    # script += "#SBATCH --time=12:00:00\n"
    # script += "#SBATCH --job-name={0}\n".format(name)
    # script += "#SBATCH --output={0}.out\n".format(os.path.join(path_to_outputs, name))
    # script += "#SBATCH --gres=gpu:1\n"
    # script += "#SBATCH --mem=24G\n"
    script += "\n"
    # script += "module load python/3.6 scipy-stack mpi4py\n"
    # script += "source /home/murphy9/.dynalearn-env/bin/activate\n"
    script += "python {0}scripts/training_script.py".format(path_to_dynalearn)
    script += " --config {0}".format(config)
    script += " --name {0}".format(name)
    script += " --num_samples {0}".format(num_samples)
    script += " --num_nodes {0}".format(num_nodes)
    script += " --resampling_time {0}".format(2)
    script += " --batch_size {0}".format(1)
    script += " --window_size {0}".format(wsize)
    script += " --window_step {0}".format(wstep)
    script += " --hide_prob {0}".format(hp)
    script += " --use_groundtruth {0}".format(1)
    script += " --tasks {0}".format(" ".join(tasks))
    script += " --metrics {0}".format(" ".join(metrics))
    script += " --to_zip {0}".format(" ".join(to_zip))
    script += " --path {0}".format(path_to_data)
    script += " --path_to_best {0}".format(path_to_best)
    script += " --path_to_summary {0}".format(path_to_summary)
    script += " --verbose 1\n"
    # script += "deactivate\n"

    path_to_script = "{0}/train-{1}.sh".format(
        os.path.join(path_to_dynalearn, "scripts/bernard/launch_scripts"), name
    )

    with open(path_to_script, "w") as f:
        f.write(script)

    os.system("bash {0}".format(path_to_script))

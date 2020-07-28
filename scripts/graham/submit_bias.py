import os
import time
from itertools import product


path_to_dynalearn = "/home/murphy9/source/dynalearn/"
path_to_dynalearn_data = (
    "/home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/"
)
path_to_all = os.path.join(path_to_dynalearn_data, "bias")
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
num_samples_array = [10000]
config_array = [
    # "sis-er",
    # "sis-ba",
    # "plancksis-er",
    "plancksis-ba",
    # "sissis-er",
    # "sissis-ba",
    # "hiddensissis-ba",
]
bias_array = np.linspace(0, 1, 11)
# tasks = ["generate_data", "train_model", "compute_metrics", "compute_summaries"]
tasks = ["generate_data", "train_model", "compute_metrics"]
metrics = ["ltp", "star-ltp"]
to_zip = [
    "config.pickle",
    # "data.h5",
    "metrics.h5",
    "history.pickle",
    "model.pt",
    "optim.pt",
]

for num_samples, config, bias in product(num_samples_array, config_array, bias_array):
    suffix = "b"str(bias)
    name = config + "-" + suffix
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
    script += " --epochs {0}".format(30)
    script += " --resampling_time {0}".format(2)
    script += " --batch_size {0}".format(1)
    script += " --use_groundtruth {0}".format(1)
    script += " --train_bias {0}".format(b)
    script += " --val_bias {0}".format(1)
    script += " --gnn_layer {0}".format("DynamicsGAT")
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

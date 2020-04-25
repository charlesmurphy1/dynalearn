import os
import time


path_to_dynalearn = "/home/murphy9/source/dynalearn/"
path_to_dynalearn_data = (
    "/home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/"
)
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
num_samples = [100, 500, 1000, 5000, 10000, 20000]
configs_to_run = [
    "sis-er",
    "sis-ba",
    "plancksis-er",
    "plancksis-ba",
    "sissis-er",
    "sissis-ba",
]

for ns in num_samples:
    for config in configs_to_run:
        suffix = "ns" + str(ns)
        name = config + "-" + suffix
        script = "#!/bin/bash\n"
        script += "#SBATCH --account=def-aallard\n"
        script += "#SBATCH --time=12:00:00\n"
        script += "#SBATCH --job-name={0}\n".format(name)
        script += "#SBATCH --output={0}.out\n".format(
            os.path.join(path_to_outputs, name)
        )
        script += "#SBATCH --gres=gpu:1\n"
        script += "#SBATCH --mem=24G\n"
        script += "\n"
        script += "module load python/3.6 scipy-stack mpi4py\n"
        script += "source /home/murphy9/.dynalearn-env/bin/activate\n"
        script += "python {0}scripts/training_script.py".format(path_to_dynalearn)
        script += " --config {0}".format(config)
        script += " --name {0}".format(name)
        script += " --num_samples {0}".format(ns)
        script += " --num_nodes {0}".format(num_nodes)
        script += " --resampling_time {0}".format(2)
        script += " --batch_size {0}".format(10)
        script += " --with_truth {0}".format(0)
        script += " --run_fast {0}".format(0)
        script += " --path {0}".format(path_to_data)
        script += " --path_to_best {0}".format(path_to_best)
        script += " --path_to_summary {0}".format(path_to_summary)
        script += " --verbose 2\n"
        script += "deactivate\n"

        seed = 0
        path_to_script = "{0}/{1}-{2}.sh".format(
            os.path.join(path_to_dynalearn, "scripts/local/launch_scripts"),
            config,
            seed,
        )

        with open(path_to_script, "w") as f:
            f.write(script)

        os.system("bash {0}".format(path_to_script))

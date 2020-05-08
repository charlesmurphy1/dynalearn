import os
import time
from itertools import product


path_to_dynalearn = (
    "/home/charles_murphy/Documents/ulaval/doctorat/projects/dynalearn-all/dynalearn/"
)
path_to_dynalearn_data = "/home/charles_murphy/Documents/ulaval/doctorat/projects/dynalearn-all/dynalearn/data"
path_to_all = os.path.join(path_to_dynalearn_data, "real-networks")
path_to_data = os.path.join(path_to_all, "full_data")
path_to_models = os.path.join(path_to_all, "models")
path_to_summary = os.path.join(path_to_all, "summary")
path_to_outputs = os.path.join(path_to_all, "outputs")

if not os.path.exists(path_to_data):
    os.makedirs(path_to_data)
if not os.path.exists(path_to_summary):
    os.makedirs(path_to_summary)
if not os.path.exists(path_to_outputs):
    os.makedirs(path_to_outputs)

edgelist_array = [
    "contacts-prox-high-school-2013",
    "copresence-InVS13",
    "copresence-InVS15",
    "copresence-LH10",
    "copresence-SFHH",
    "copresence-Thiers13",
    "ia-contacts_dublin",
    "ia-contacts_hypertext2009",
]

config_array = [
    ("approx","sis"),
    ("semi-exact","sis"),
    ("approx","plancksis"),
    ("semi-exact","plancksis"),
]

for edgelist, config in product(edgelist_array, config_array):
    name = config[0] + "-" + config[1] + "-" + edgelist
    path_to_edgelist = os.path.join(path_to_all, "datasets", edgelist, "edgelist.txt")
    path_to_model = os.path.join(path_to_models, config[0])
    script = "#!/bin/bash\n"
    # script += "#SBATCH --account=def-aallard\n"
    # script += "#SBATCH --time=12:00:00\n"
    # script += "#SBATCH --job-name={0}\n".format(name)
    # script += "#SBATCH --output={0}.out\n".format(os.path.join(path_to_outputs, name))
    # script += "#SBATCH --gres=gpu:1\n"
    # script += "#SBATCH --mem=24G\n"
    # script += "\n"
    # script += "module load python/3.6 scipy-stack mpi4py\n"
    # script += "source /home/murphy9/.dynalearn-env/bin/activate\n"
    script += "python {0}scripts/realnet_script.py".format(path_to_dynalearn)
    script += " --config {0}".format(config[1])
    script += " --name {0}".format(name)
    script += " --path_to_data {0}".format(path_to_data)
    script += " --path_to_edgelist {0}".format(path_to_edgelist)
    script += " --path_to_model {0}".format(path_to_model)
    script += " --path_to_summary {0}".format(path_to_summary)
    script += " --verbose 1\n"
    # script += "deactivate\n"

    seed = 0
    path_to_script = "{0}/{1}.sh".format(
        os.path.join(path_to_dynalearn, "scripts/hector/launch_scripts"), name
    )

    with open(path_to_script, "w") as f:
        f.write(script)

    os.system("bash {0}".format(path_to_script))

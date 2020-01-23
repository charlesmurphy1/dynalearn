import dynalearn as dl


config = dl.ExperimentConfig.sis_er()
config.save("../data/training/")
config = dl.ExperimentConfig.sis_ba()
config.save("../data/training/")

config = dl.ExperimentConfig.plancksis_er()
config.save("../data/training/")
config = dl.ExperimentConfig.plancksis_ba()
config.save("../data/training/")

config = dl.ExperimentConfig.sissis_er()
config.save("../data/training/")
config = dl.ExperimentConfig.sissis_ba()
config.save("../data/training/")


script  = "#!/bin/bash\n"
script += "#SBATCH --account=def-aallard\n"
script += "#SBATCH --time=24:00:00\n"
script += "#SBATCH --job-name=training_{0}\n".format(config.config["name"])
script += "#SBATCH --output={}.out\n".format(config.config["name"])
script += "#SBATCH --gres=gpu:1\n"
script += "#SBATCH --cpus-per-task=4\n"
script += "#SBATCH --mem=4000M\n"
script += "\n\n"
script += "source ~/pyenv/"
script += ""
script += ""
script += ""
script += ""
script += ""
script += ""
script += ""
script += ""

#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --job-name=test
#SBATCH --time=00:01:00
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=4000M               # memory per node
#SBATCH --time=0-03:00            # time (DD-HH:MM)
echo 'Hello, world!

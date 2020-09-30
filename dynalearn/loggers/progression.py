from datetime import datetime

from .logger import Logger


class ProgressionLogger(Logger):
    def __init__(self, path_to_log="./progress"):
        self.path_to_log = path_to_log
        self.step = 0
        self.stepname = None
        Logger.__init__(self)

    def on_task_begin(self):
        _file = open(self.path_to_log, "w")

        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log["begin"] = time
        _file.write(f"begin: {time}\n")
        _file.close()

    def on_task_midstep(self, stepname=None):
        _file = open(self.path_to_log, "a")
        if self.stepname != stepname:
            self.stepname = stepname
            self.step = 0

        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log[f"{self.stepname}-{self.step}"] = time
        _file.write(f"{self.stepname}-{self.step}: {time}\n")
        _file.close()
        self.step += 1

    def on_task_end(self):
        _file = open(self.path_to_log, "a")
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log["end"] = time
        _file.write(f"end: {time}\n")
        _file.close()

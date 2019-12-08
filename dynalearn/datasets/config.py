class GeneratorConfig:
    @classmethod
    def default(cls):

        cls = cls()

        cls.batch_size = -1
        cls.resampling_time = 2
        cls.max_null_iter = 100
        cls.shuffle = True
        cls.with_truth = False

        return cls

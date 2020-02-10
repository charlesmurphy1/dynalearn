class GNNConfig:
    @classmethod
    def SISGNN(cls):

        cls = cls()

        cls.num_states = 2
        cls.in_features = [32]
        cls.attn_features = [32]
        cls.out_features = [32]
        cls.n_heads = [1]
        cls.in_activation = "relu"
        cls.attn_activation = "relu"
        cls.out_activation = "relu"

        cls.weight_decay = 0.0001
        cls.tf_seed = 2

        return cls

    @classmethod
    def ComplexSISGNN(cls):

        cls = cls()

        cls.num_states = 2
        cls.in_features = [32]
        cls.attn_features = [32]
        cls.out_features = [32]
        cls.n_heads = [1]
        cls.in_activation = "relu"
        cls.attn_activation = "relu"
        cls.out_activation = "relu"

        cls.weight_decay = 0.0001
        cls.tf_seed = 2

        return cls

    @classmethod
    def SIRGNN(cls):

        cls = cls()

        cls.num_states = 3
        cls.in_features = [32, 32]
        cls.attn_features = [32]
        cls.out_features = [32, 32]
        cls.n_heads = [2]
        cls.in_activation = "relu"
        cls.attn_activation = "relu"
        cls.out_activation = "relu"

        cls.weight_decay = 0.0001
        cls.tf_seed = 2

        return cls

    @classmethod
    def ComplexSIRGNN(cls):

        cls = cls()

        cls.num_states = 3
        cls.in_features = [32, 32]
        cls.attn_features = [32]
        cls.out_features = [32, 32]
        cls.n_heads = [2]
        cls.in_activation = "relu"
        cls.attn_activation = "relu"
        cls.out_activation = "relu"

        cls.weight_decay = 0.0001
        cls.tf_seed = 2

        return cls

    @classmethod
    def SISSISGNN(cls):

        cls = cls()

        cls.num_states = 4
        cls.in_features = [64, 64]
        cls.attn_features = [64]
        cls.out_features = [64, 64]
        cls.n_heads = [4]
        cls.in_activation = "relu"
        cls.attn_activation = "relu"
        cls.out_activation = "relu"

        cls.weight_decay = 0.0001
        cls.tf_seed = 2

        return cls

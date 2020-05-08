from .config import Config


class SummariesConfig(Config):
    @classmethod
    def sis_fast(cls):
        cls = cls()
        cls.names = [
            "TrueLTPSummary",
            "GNNLTPSummary",
            "MLELTPSummary",
            "TrueGNNStarJSDErrorSummary",
            "TrueMLEJSDErrorSummary",
            "TrueUniformStarJSDErrorSummary",
            "StatisticsSummary",
        ]
        cls.axis = 1
        cls.err_reduce = "percentile"
        cls.transitions = [(0, 1), (1, 0)]
        return cls

    @classmethod
    def sis_complete(cls):
        cls = cls()
        cls.names = [
            "TrueLTPSummary",
            "GNNLTPSummary",
            "MLELTPSummary",
            "TrueGNNStarJSDErrorSummary",
            "TrueMLEJSDErrorSummary",
            "TrueUniformStarJSDErrorSummary",
            "StatisticsSummary",
            "TruePESSSummary",
            "GNNPESSSummary",
            "TruePEMFSummary",
            "GNNPEMFSummary",
        ]
        cls.axis = 1
        cls.err_reduce = "percentile"
        cls.transitions = [(0, 1), (1, 0)]
        return cls

    @classmethod
    def plancksis_fast(cls):
        cls = cls()
        cls.names = [
            "TrueLTPSummary",
            "GNNLTPSummary",
            "MLELTPSummary",
            "TrueGNNStarJSDErrorSummary",
            "TrueMLEJSDErrorSummary",
            "TrueUniformStarJSDErrorSummary",
            "StatisticsSummary",
        ]
        cls.axis = 1
        cls.err_reduce = "percentile"
        cls.transitions = [(0, 1), (1, 0)]
        return cls

    @classmethod
    def plancksis_complete(cls):
        cls = cls()
        cls.names = [
            "TrueLTPSummary",
            "GNNLTPSummary",
            "MLELTPSummary",
            "TrueGNNStarJSDErrorSummary",
            "TrueMLEJSDErrorSummary",
            "TrueUniformStarJSDErrorSummary",
            "StatisticsSummary",
            "TruePESSSummary",
            "GNNPESSSummary",
            "TruePEMFSummary",
            "GNNPEMFSummary",
        ]
        cls.axis = 1
        cls.err_reduce = "percentile"
        cls.transitions = [(0, 1), (1, 0)]
        return cls

    @classmethod
    def sissis_fast(cls):
        cls = cls()
        cls.names = [
            "TrueGNNStarJSDErrorSummary",
            "TrueMLEJSDErrorSummary",
            "TrueUniformStarJSDErrorSummary",
            "StatisticsSummary",
        ]
        cls.axis = 1
        cls.err_reduce = "percentile"
        return cls

    @classmethod
    def sissis_complete(cls):
        cls = cls()
        cls.names = [
            "TrueGNNStarJSDErrorSummary",
            "TrueMLEJSDErrorSummary",
            "TrueUniformStarJSDErrorSummary",
            "StatisticsSummary",
            "TruePESSSummary",
            "GNNPESSSummary",
            "TruePEMFSummary",
            "GNNPEMFSummary",
        ]
        cls.axis = 1
        cls.err_reduce = "percentile"
        return cls

    @classmethod
    def rtn_forecast(cls):
        cls = cls()
        cls.names = [
            "TrueSSSummary",
            "GNNSSSummary",
            "TrueRTNForecastSummary",
            "GNNRTNForecastSummary",
        ]
        return cls

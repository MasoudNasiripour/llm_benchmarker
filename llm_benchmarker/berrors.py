class DownloaderFunctionWantFoundError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)


class IncompleteModelInfoError(Exception):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)


class LengthMisMatchError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)


class InvalidPredictionsForBenchmarkError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)


class MetricCalculationError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
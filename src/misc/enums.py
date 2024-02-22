from enum import Enum, auto


class Mode(Enum):
    PREDICT = auto()
    TRAIN = auto()
    OPTIMIZE = auto()


class ModelType(Enum):
    GCNCONV = auto()
    SAGECONV = auto()
    SAGECONVMLP = auto()
    SAGECONVGATMLP = auto()


class LabelType(Enum):
    METAL = auto()
    SMALL = auto()
    NUCLEAR = auto()

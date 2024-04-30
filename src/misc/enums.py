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
    METAL = 0
    NUCLEAR = 1
    SMALL = 2


class DSSPStructure(Enum):
    HELIXALPHA = 0
    BETABRIDGE = 1
    EXTENDEDSRTRAND = 2
    HELIX3_10= 3
    HELIXPI = 4
    HELIXK = 5
    TURN = 6
    BEND = 7


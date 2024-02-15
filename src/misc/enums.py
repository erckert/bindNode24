from enum import Enum, auto


class Mode(Enum):
    PREDICT = auto()
    TRAIN = auto()
    OPTIMIZE = auto()


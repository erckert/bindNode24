from pathlib import Path
from config import App
from misc.enums import Mode
from machine_learning.predict import run_prediction
from machine_learning.train import run_optimization, run_training

from setup.configProcessor import select_mode_from_config

if __name__ == "__main__":
    print('I am running')
    mode = select_mode_from_config()
    print(f'Selected mode is {mode.name}')
    match mode:
        case Mode.PREDICT:
            run_prediction()
        case Mode.OPTIMIZE:
            run_optimization()
        case Mode.TRAIN:
            run_training()

from misc.enums import Mode
from misc.Cache import setup_chache
from machine_learning.predict import run_prediction
from machine_learning.train import run_optimization, run_training

from setup.configProcessor import validate_config, select_mode_from_config, use_cache, get_model_parameter_dict
from setup.generalSetup import seed_all

from output.outputFileWriting import write_predictions_to_file

if __name__ == "__main__":
    print('I am running')
    seed_all(seed=42)
    validate_config()
    mode = select_mode_from_config()
    print(f'Selected mode is {mode.name}')
    if use_cache():
        setup_chache()
    match mode:
        case Mode.PREDICT:
            model_parameters = get_model_parameter_dict(only_first_value=True)
            predictions = run_prediction(model_parameters)
            write_predictions_to_file(predictions)
        case Mode.OPTIMIZE:
            model_parameters = get_model_parameter_dict(only_first_value=False)
            run_optimization(model_parameters)
        case Mode.TRAIN:
            model_parameters = get_model_parameter_dict(only_first_value=True)
            run_training(model_parameters)

import wandb
from datetime import datetime

LOGGING=False
def init_log_run_to_wandb(params):
    """params is a dictionary of hyperparams"""
    # create a unique run name to log
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M")
    run_name=f"{params['architecture']}_{params['features']}_feat_drop{params['dropout']}_cutoff{params['cutoff']}_{date_time}"
    wandb.init(
        # set the wandb project where this run will be logged
        project="bindNode23",
        
        # track hyperparameters and run metadata
        config=params,
        name=run_name
    )


def init_final_log_to_wandb(params):
    """params is a dictionary of hyperparams"""
    # create a unique run name to log
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M")
    run_name=f"{params['architecture']}_{params['features']}_feat_drop{params['dropout']}_cutoff{params['cutoff']}_{date_time}"
    wandb.init(
        # set the wandb project where this run will be logged
        project="bindNode23_CVResults",
        
        # track hyperparameters and run metadata
        config=params,
        name=run_name
    )

def log_epoch_metrics_to_wandb(metrics):
    """metrics is a dictionary of metrics"""
    # log epoch metrics to wandb
    wandb.log(metrics)

def log_final_metrics_to_wandb(metrics, params):
    """metrics is a nested dictionary of metrics. need to init a run cause it's outside CV"""
    # log final metrics to wandb by prepending Test to keys
    init_final_log_to_wandb(params=params)
    wandb.log(metrics)
    finish_run_in_wandb()

def finish_run_in_wandb():
    wandb.finish()


import os
import pathlib
from datetime import datetime


def str_current_time():
    """
    Returns the current time in readable string.
    """
    now = datetime.now()
    current_datetime = \
        str(now.year) + \
        str('%02d' % now.month) + \
        str('%02d' % now.day) + \
        str('%02d' % now.hour) + \
        str(now.strftime('%M')) + \
        str(now.strftime('%S'))

    return current_datetime


def log(str, logfile=None, verbose=True):
    """
    Prints the provided string, and also logs it if a logfile is passed.
    Parameters
    ----------
    str: str
        String to be printed/logged.
    logfile: str (optional)
        File to log into.
    verbose: bool (optional)
        Prints log if verbose is true.
    """
    str = f'[{datetime.now()}] {str}'
    if verbose:
        print(str)
    if logfile is not None:
        # file existence
        if not os.path.isfile(logfile):
            with open(logfile, mode='w') as f:
                print(str, file=f)
                f.close()
        else:
            with open(logfile, mode='a') as f:
                print(str, file=f)


class Scheduler:
    """
    Custom scheduler class that early-stops the optimization process
    Parameters
    ----------
    patience: int
        a number of bad epochs that can be endured until early-stop
    max_epoch: int
        maximum number of epoch for training.
    save_dir: str
        saving directory for the parameters
    """
    def __init__(self, patience=10, max_epoch=100, save_dir=None):
        self.save_dir = save_dir
        self.patience = patience
        self.max_epoch = max_epoch
        self.last_epoch = 0
        self.num_bad_epoch = 0
        self.best_loss = 1e15
        self.stop = False

    def step(self, loss):
        """
        calculates a number of bad epochs and checks tolerences.
        Parameters
        ----------
        loss: float
        """
        self.last_epoch += 1
        if self.last_epoch > self.max_epoch:
            self.stop = True

        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epoch = 0
        else:
            self.num_bad_epoch += 1


def enable_dropout(model):
    """
    Enables dropout of the given model.
    Used to produce predictive distribution of the model
    and obtain model uncertanties.
    Parameters
    ----------
    model: nn.Module
    """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

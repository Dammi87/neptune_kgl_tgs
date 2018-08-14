import numpy as np
import logging
from src.lib.neptune import get_params

params = get_params()

class GlobalStep(object):

    the_one = None

    def __new__(self):
        if self.the_one is None:
            self.the_one = super(GlobalStep, self).__new__(self)
            self.the_one.current = 0

        return self.the_one



def convert_step_to_epoch(global_step):
    """Convert global_step to floating point number of epochs.

    Parameters
    ----------
    global_step : Tensor
        Current global training step
    """
    steps_per_epoch = params.dataset_size * (1 - params.validation_split)

    return float(global_step) / steps_per_epoch


class EarlyStopping():
    """Simple early stopping class."""

    def __init__(self, start_epoch=2, max_events=10, maximize=False):
        """Initialize class.

        Parameters
        ----------
        start_epoch : int, optional
            How many times add should be called, until it get's activated
        max_events : int, optional
            Maximum number of non-better events until early stopping is signalled
        """
        self._add_calls = 0
        self._start_epoch = start_epoch
        self._n_events = 0
        self._max_events = max_events
        self._seen_losses = []
        self._best_loss = np.inf
        if maximize:
            self._best_loss = -np.inf
        self._early_stop = False
        self._maximize = maximize
        # Need to use the tensorflow logger
        self._logging = logging.getLogger('tensorflow')

    def _is_improvement(self, current_metric):
        """Check if current metric is better or not."""
        if self._maximize:
            return self._best_loss <= current_metric
        else:
            return self._best_loss >= current_metric

    def add(self, current_metric):
        """Add current loss.

        Parameters
        ----------
        current_metric : TYPE
            Current loss metric
        """
        # Add has been called
        self._add_calls += 1

        if self._is_improvement(current_metric):
            self._best_loss = current_metric
            self._n_events = 0
            self._logging.info('EarlyStopping: Improved, current best %1.4f' % self._best_loss)
        else:
            self._n_events += 1
            self._logging.info('EarlyStopping: No improvements, current best %1.4f' % self._best_loss)

        if self._n_events == self._max_events:
            self._logging.info('EarlyStopping: Early stopping condition reached')
            self._early_stop = True

    def _is_activated(self):
        """Return true if EarlyStopping should be activated."""
        return self._add_calls >= self._start_epoch

    def should_stop(self):
        """Return true if early stopping criteria is reached."""
        return self._early_stop and self._is_activated()

    def is_better(self):
        """Return true if current loss is the best seen."""
        return self._n_events == 0

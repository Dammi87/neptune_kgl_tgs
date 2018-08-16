import numpy as np
import logging
from src.lib.neptune import get_params

params = get_params()


class Logger(object):
    """This class can be used to share information about training."""
    the_one = None

    def __new__(cls):
        """Initialize as a singleton."""
        if cls.the_one is None:
            cls.the_one = super(Logger, cls).__new__(cls)
            cls.the_one.current_global_step = 0
            cls.the_one.current_epoch = 0
            cls.the_one.current_lr = 0
            cls.the_one.current_loss = 0
            cls.the_one.current_no_improvements = 0

        return cls.the_one

    def log_global_step(self, step):
        """Set global step."""
        self.current_global_step = step

    def log_epoch(self, epoch):
        """Set epoch."""
        self.current_epoch = epoch

    def log_learning_rate(self, lr):
        """Set learning rate."""
        self.current_lr = lr

    def log_loss(self, loss):
        """Set loss."""
        self.current_loss = loss

    def log_no_improvements(self, N):
        """Set how many non-improvements have happened."""
        self.current_no_improvements = N

    def get_global_step(self):
        """Get global step."""
        return self.current_global_step

    def get_epoch(self):
        """Get epoch."""
        return self.current_epoch

    def get_learning_rate(self):
        """Get learning rate."""
        return self.current_lr

    def get_loss(self):
        """Get loss."""
        return self.current_loss

    def get_no_improvements(self):
        """Get how many non improvements have happened."""
        return self.current_no_improvements


class AdjustLearningRate():
    """Class for adjusting the learning rate according to specific settings."""

    def __init__(self):
        """Initialize class."""
        self._logger = Logger()
        self._logging = logging.getLogger('tensorflow')
        self._params = get_params()

    def _basic(self):
        """Simply return the default learning rate."""
        self._logging.info('AdjustLearningRate._basic: No change')
        return params.learning_rate

    def _reduce_on_plateau(self):
        """If plateaued, reduce learning rate by specified ratio."""
        nbr = self._logger.get_no_improvements()
        if nbr >= self._params.lr_patience:
            lr = self._logger.get_learning_rate() * self._params.lr_plateau_reduce
            lr = max(self._params.lr_plateau_min_lr, lr)  # Minimum safeguard
            self._logger.log_learning_rate(lr)
            self._logging.info('AdjustLearningRate._reduce_on_plateau: Learning rate changed to %1.6f' % lr)
            return lr
        self._logging.info('AdjustLearningRate._reduce_on_plateau: No change, count at %d' % nbr)
        return self._logger.get_learning_rate()

    def get_lr(self):
        """Return the learning rate."""
        method = getattr(self, '_%s' % self._params.lr_type)
        return method()


class AdjustLayerFreeze():
    """Class for adjusting what layers should be active during training.

    This is usefull when it's desired to freeze for example the encoder part of a
    network that has pre-trained weights, allowing the decoder to be trained more
    fully. Then at later stages, activate the encoder training.
    """

    def __init__(self):
        """Initialize class."""
        self._logger = Logger()
        self._logging = logging.getLogger('tensorflow')
        self._params = get_params()

    def _none(self):
        """No freezing."""
        self._logging.info('AdjustLayerFreeze._basic: Nothing applied, basic learning')
        return None, True

    def _on_epoch(self):
        """Here you can define settings based on network type.

        The settings are
            NETWORK_NAME
                'basic': Layers to activate when epoch is not reached
                'at_epoch': Layers to activate when epoch is reached

        NOTE: Once epoch is reached, it won't revert back to basic!
              These settings are also only activated IF warm_start is TRUE
        """
        settings_cfg = {
            'vgg_16_unet':
                {
                    'basic': ['Decoder'],
                    'at_epoch': ['Encoder', 'vgg_16']
                },
            'resnet_152':
                {
                    'basic': ['Decoder'],
                    'at_epoch': ['Decoder', 'resnet_v2_152']
                },
            'basic':
                {
                    'basic': [None],
                    'at_epoch': [None]
                }
        }
        activiated = [True, False]  # Basic version activated first
        setting_type = 'basic'
        if self._params.warm_start:
            setting_type = self._params.network_type

        # If no warm start, simply return the basic one
        if setting_type == 'basic':
            return self._none()

        # Otherwise, we can continue
        settings = settings_cfg[setting_type]

        nbr = self._logger.get_epoch()
        settings = settings_cfg[setting_type]
        if nbr >= self._params.freeze_until_epoch:
            self._logging.info('AdjustLayerFreeze._varlist_on_epoch: training {}'.format(settings['at_epoch']))
            activiated = [False, True]  # Activate both var lists
        else:
            self._logging.info('AdjustLayerFreeze._varlist_on_epoch: training {}'.format(settings['basic']))
            activiated = [True, False]  # Basic only

        return settings, activiated

    def get_collection_list(self):
        """Return the learning rate."""
        method = getattr(self, '_%s' % self._params.layer_freeze_type)
        return method()


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

    def get_nbr_non_improvements_in_row(self):
        """Return how many times, the metric has NOT improved."""
        return self._n_events

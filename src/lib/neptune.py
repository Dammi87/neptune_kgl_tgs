"""Contains methods that help with neptune related stuff."""
import tensorflow as tf
from deepsense import neptune
import argparse
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training import training_util


# Get neptune parameters
ctx = neptune.Context()

def get_job_id():
    """Fetches the neptune job id."""
    return ctx.experiment_id


def get_params(as_dict=False):
    """Fetch the neptune parameters, "None" are set to None."""
    new_args = {}
    for key in ctx.params:
        if ctx.params[key] == "None":
            new_args[key] = None
        else:
            new_args[key] = ctx.params[key]

    if as_dict:
        return new_args

    return argparse.Namespace(**new_args)


def get_metric():
    """Return metric if available."""
    if ctx.metric is None:
        return None, None

    if ctx.metric.direction == 'minimize':
        def op(old, new):
            return min(old, new)
    else:
        def op(old, new):
            return max(old, new)

    return ctx.metric.channel_name, op


class NeptuneCollector():
    """Neptune summary hook."""

    def __init__(self):
        """Initialize a Neptune data collector."""
        self._ctx = ctx
        self._metrics = {}
        self._metric_cfg = get_metric()

    def _maybe_add(self, metric_name, metric_value):
        """Only adds the metric to the internal dictionary if it's not there already."""
        if metric_name not in self._metrics:
            self._metrics[metric_name] = metric_value

    def send(self, metrics):
        """Send metric results.

        Parameters
        ----------
        metrics : Dict
            Output from estimator.evaluate
        """
        # Pop out the global step
        global_step = metrics.pop('global_step')

        for name in metrics:
            value = metrics[name]
            self._maybe_add(name, value)

            # If this tensorflow metric is a metric in Neptune, determine if
            # it should be updated or not
            if name == self._metric_cfg[0]:
                self._metrics[name] = self._metric_cfg[1](self._metrics[name], value)
            else:
                self._metrics[name] = value

        # After metrics have been correctly allocated, send them to neptune
        for name in self._metrics:
            value = self._metrics[name]
            self._ctx.channel_send(name, x=global_step, y=value)

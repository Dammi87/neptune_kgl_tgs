"""Loads all architectures, returns the desired one."""
from src.lib.neptune import get_params
import os
import importlib

def get_modules():
    path = os.path.dirname(__file__)
    modules = os.listdir(path)

    # Remove non-modules
    modules = [module for module in modules if not module.startswith('__')]
    modules = [module for module in modules if not '.py'in module]

    return modules


def load_architectures():
    return [importlib.import_module('.%s' % module, __package__) for module in get_modules()]

__all__ = load_architectures()

# __all__ = [importlib.import_module('.%s' % filename, __package__) for filename in [os.path.splitext(i)[0] for i in os.listdir(os.path.dirname(__file__)) if os.path.splitext(i)[1] in pyfile_extes] if not filename.startswith('__')]

# Method to get specific architecture
def get_network():
    network_type = get_params().network_type
    if network_type not in get_modules():
        raise Exception('Network type {}, not found, available types are: {}'.format(network_type, get_modules()))

    module = importlib.import_module('.%s' % network_type, __package__)
    return getattr(module, 'network')


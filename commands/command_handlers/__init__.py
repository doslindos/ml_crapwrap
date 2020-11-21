import argparse
from utils.modules import if_callable_class_function, get_callable_class_functions
from ..command_functions.utils import validate_args, GPU_config

def create_args(parser_args, add_args):
    #Creates Argument parser arguments
    # In:
    #   parser_args:                dict, Argument parser input
    #   add_args:                   dict, Argument parser add_argument method inputs
    # Out:
    #   parser.parse_args()         parsed arguments

    parser = argparse.ArgumentParser(**parser_args)
    for add in add_args:
        name = add.pop('name')
        parser.add_argument(*name, **add)
    return parser.parse_args()

import main
from argparse import Namespace
import traceback
from .env_test_functions import create_dataset_test, training_test, testing_test
from .mnist_tests import run_mnist
from .spotify_tests import run_spoti

def print_loop(test_dict):
    for key, value in test_dict.items():
        if isinstance(value, str):
            #print(key, " : ", value)
            print("{:<40} {:<5}".format(key, value))
        elif isinstance(value, dict):
            print("\n", "   "+key+": ")
            print_loop(value)
        else:
            print("Unknown ", value)


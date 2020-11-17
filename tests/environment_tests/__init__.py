from argparse import Namespace
import traceback

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


from sys import argv, exit
from commands.command_handlers.function_test import FunctionTestArgs
from utils.modules import get_callable_class_functions

if __name__ == '__main__':
    
    # Get all callable arguments
    callable_functions = get_callable_class_functions(FunctionTestArgs)
    
    # Check for arguments
    if len(argv) < 2:
        print("You have to specify which function is used\nUsable command ", callable_functions)
        exit()

    # First argument as "main" command
    maincommand = argv[1]
    # If main command is usable call it
    if maincommand in callable_functions:
        getattr(FunctionTestArgs, maincommand)()
    else:
        print("Command is not recognized. \nUsable commands ",callable_functions," example: python test.py test_model [options]")
    

from sys import argv, exit
from commands.command_handlers.create import CreateArgs
from utils.main_utils import get_callable_class_functions

if __name__ == '__main__':
    
    # Get all callable arguments
    callable_functions = get_callable_class_functions(CreateArgs)
    
    # Check for arguments
    if len(argv) < 2:
        print("You have to specify which function is used\nUsable command ", callable_functions)
        exit()

    # first argument as "main" command
    maincommand = argv[1]
    # if main command is usable call it
    if maincommand in callable_functions:
        getattr(CreateArgs, maincommand)()
    else:
        print("Command is not recognized. \nUsable commands ",callable_functions," example: python create.py dataset [options]")

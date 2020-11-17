from sys import argv, exit
from commands.command_handlers.test import TestArgs
from utils.main_utils import get_callable_class_functions

if __name__ == '__main__':
    
    # Get all callable arguments
    callable_functions = get_callable_class_functions(TestArgs)
    
    # Check for arguments
    if len(argv) < 2:
        print("You have to specify which function is used\nUsable command ", callable_functions)
        exit()

    # First argument as "main" command
    maincommand = argv[1]
    # If main command is usable call it
    if maincommand in callable_functions:
        getattr(TestArgs, maincommand)()
    else:
        print("Command is not recognized. \nUsable commands ",callable_functions," example: python test.py test_model [options]")
    
 #   log_dict = {
 #           'Create datasets':{},
 #           'Train datasets':{},
 #           'Test datasets':{}
 #           }

 #   if maincommand == 'command_test':
 #       log_dict = EnvironmentTester(parsed_args)
 #   if tests in ['mnist', 'all']:
 #       log_dict = tests.environment_tests.run_mnist(log_dict)
    
 #   if tests in ['spotify', 'all']:
#        log_dict = tests.environment_tests.run_spoti(log_dict)

 #   tests.environment_tests.print_loop(log_dict)

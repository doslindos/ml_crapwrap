from sys import argv, exit
from commands import args, commandline_functions

if __name__ == '__main__':
    
    # Check for arguments
    if len(argv) < 2:
        commandline_functions.error()

    # First argument as "main" command
    maincommand = argv[1]
    # Check if the main command is valid
    if maincommand in dir(args):
        parsed_args = getattr(args, maincommand)()
    else:
        error("Command is not recognized. \nUsable commands 'dataset', 'data', 'train', 'test', 'plot' example: python main.py dataset [options]")
    
    # Validate input arguments
    commandline_functions.validate_args(vars(parsed_args))
    
    #Set up gpu if you use GPU version of the tensorflow
    commandline_functions.GPU_config()
    
    # Dataset creation
    if maincommand == 'dataset':
        commandline_functions.create_dataset(parsed_args)
    # Data inspection
    elif maincommand == 'data':
        commandline_functions.dataset_information(parsed_args)
    # Train a model
    elif maincommand == 'train':
        commandline_functions.train_model(parsed_args)
    # Test a model
    elif maincommand == 'test':
        commandline_functions.test_model(parsed_args)
    # Model plotting
    elif maincommand == 'plot':
        commandline_functions.plot_model(parsed_args)

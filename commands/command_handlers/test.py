from . import create_args, if_callable_class_function, get_callable_class_functions
from .create import CreateArgs
from ..command_functions.plot import plot_model
from ..command_functions.test import test_model
from importlib.util import find_spec
from importlib import import_module
import traceback

class TestArgs:
    
    def data():
        #Defines data information function inputs and calls argument parser
        parser_args = {'description':'Show data information'}
        add_args = [
            {'name':['command'], 'type':str, 'help':'Main command'},
            {'name':['-d'], 'type':str, 'required':True, 'help':'Name of the dataset'},
            {'name':['-info'], 'type':str, 'required':True, 'help':'Name of the information function'},
            {'name':['-l'], 'type':str, 'required':True, 'help':'Name of the data key'},
            {'name':['--merge_key'], 'type':str, 'default':None, 'help':'Name of the merge key'},
            ]
        # Parse arguments
        parsed_args = create_args(parser_args, add_args)
        # Validate input arguments
        validate_args(vars(parsed_args))
        # Run function
        dataset_information(parsed_args)
    
    def test_model():
        #Defines test function inputs and calls argument parser
        parser_args = {'description':'Test a model'}
        add_args = [
            {'name':['command'], 'type':str, 'help':'Main command'},
            {'name':['-d'], 'type':str, 'required':True, 'help':'Dataset name'},
            {'name':['-pf'], 'type':str, 'default':None, 'help':'Name for the preprocessing function'},
            {'name':['-m'], 'type':str, 'default':"NeuralNetworks", 'help':'Model name'},
            {'name':['-test'], 'type':str, 'required':True, 'help':'Test type'},
            {'name':['-t'], 'type':int, 'default':None, 'help':'Test set size'},
            {'name':['--dataset_type'], 'type':str, 'default':'test', 'help':'Dataset type to be used'},
            {'name':['--sub_sample'], 'type':int, 'default':None, 'help':'Use a subsample of the dataset'}
            ]
        
        # Parse arguments
        parsed_args = create_args(parser_args, add_args)
        # Validate input arguments
        validate_args(vars(parsed_args))
        # Set up gpu if you use gpu version of the tensorflow
        GPU_config()
        # Run function
        test_model(parsed_args)

    def plot():
        #Defines plotting function inputs and calls argument parser
        parser_args = {'description':'Plot models outputs'}
        add_args = [
            {'name':['command'], 'type':str, 'help':'Main command'},
            {'name':['-d'], 'type':str, 'required':True, 'help':'Dataset name'},
            {'name':['-m'], 'type':str, 'default':"NN", 'help':'Model name'},
            {'name':['-plot'], 'type':str, 'required':True, 'help':'Plot type'},
            {'name':['--dataset_type'], 'type':str, 'default':'test', 'help':'Dataset type to be used'},
            {'name':['--plot_dims'], 'type':int, 'default':2, 'help':'Plot dimensions'},
            {'name':['--function'], 'type':str, 'default':'PCA', 'help':'Dimensionality reduction function. From sklearn.decomposition'},
            ]
        
        # Parse arguments
        parsed_args = create_args(parser_args, add_args)
        # Validate input arguments
        validate_args(vars(parsed_args))
        # Run function
        plot_model(parsed_args)


    def create_command_test():
        
        conf_loaction = "tests.environment_tests.test_confs."

        def run(command, data):

            # Check if command to be tested is in main commands
            if if_callable_class_function(CreateArgs, command):
                
                # Get test configuration
                if find_spec(conf_loaction+data) is not None:
                    
                    # Get the configuration file
                    arg_conf_file = import_module(conf_loaction+ data)
                    
                    # Get the right configuration
                    if hasattr(arg_conf_file, command):
                    
                        arg_conf = getattr(arg_conf_file, command)
                        # Finally run the function
                        try:
                            getattr(CreateArgs, command)(arg_conf)
                            print("Function "+command+" with "+data+" works...")
                        except Exception as e:
                            print(traceback.print_exception(e))
                            print(e)
                            print("Function "+command+" with "+data+" doesn't work!")

                    
                    else:
                        print("The configuration file: ", data, " did not have a configuration called: ", command)

                else:
                    print("No configuration found for ", data)
            
            else:
                print("Command not found... \nUsable commands to test: ",get_callable_class_functions(CreateArgs)," ")
        

        #Defines test function inputs and calls argument parser
        parser_args = {'description':'Test a creating command (dataset and model creation)'}
        
        add_args = [
            {'name':['command'], 'type':str, 'help':'Main command'},
            {'name':['-test_command'], 'type':str, 'required':True, 'help':'Name of the command to be tested, use "all" to test all commands'},
            {'name':['-testing_data'], 'type':str, 'default':'mnist', 'help':'Data used for testing. "all" uses every dataset configured in tests/environment_tests/test_confs/'}
            ]
        
        # Parse arguments
        parsed_args = create_args(parser_args, add_args)
        # Check if command to be tested is in main commands
        if parsed_args.test_command != 'all':
            if parsed_args.testing_data != 'all':
                run(parsed_args.test_command, parsed_args.testing_data)
            else:
                print("Command test call with multiple test datasets isn't implemented yet...")
                exit()

        # Run all tests
        else:
            for command in get_callable_class_functions(CreateArgs):
                if parsed_args.testing_data != 'all':
                    run(command, parsed_args.testing_data)
                else:
                    for data_confs in get_callable_class_functions(test_confs):
                        print("Command test call with multiple test datasets isn't implemented yet...")
                
    

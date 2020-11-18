from . import create_args, if_callable_class_function, get_callable_class_functions
from .create import CreateArgs
import models
from third_party.tensorflow.test import util
from importlib.util import find_spec
from importlib import import_module
import traceback

class FunctionTestArgs:
    
    def test_create_command():
        
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
            # TODO Rework configurations folder location to get confs by name for example mnist
            #{'name':['-testing_model'], 'type':str, 'default':'NeuralNetworks', 'help':'Data used for testing. "all" tests every configured model in models'}
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

    def test_tensorflow_model_functions():

        #Defines test function inputs and calls argument parser
        parser_args = {'description':'Test a model with fake data'}
        
        add_args = [
            {'name':['command'], 'type':str, 'help':'Main command'},
            {'name':['-test_model'], 'type':str, 'required':True, 'help':'Name of the model to be tested'},
            {'name':['-test_configuration'], 'type':str, 'required':True, 'help':'Name of the model configuration to be tested'},
            {'name':['-test_type'], 'type':str, 'required':True, 'help':'Name of the type of test'},
            
            ]
        
        # Parse arguments
        parsed_args = create_args(parser_args, add_args)
        # Create a model
        model_handler = models.ModelHandler(None, parsed_args.test_model, parsed_args.test_configuration)
        model = model_handler.model
        
        # Get data
        conf_loaction = "tests.environment_tests.test_confs."
        test_data_creator = import_module('tests.environment_tests.test_data.create')
        inshape = model.c['input_shape'].copy()
        inshape.insert(0, 1)
        x1 = test_data_creator.create_numpy_data((inshape), 0.001, model.c['data_type'])
        x2 = test_data_creator.create_numpy_data((inshape), 1, model.c['data_type'])
        x3 = test_data_creator.create_numpy_data((inshape), 0.01, model.c['data_type'])
        
        t = parsed_args.test_type
        
        try:
            # Run the model
            initial_output1 = model.run(x1)
            initial_output2 = model.run(x2)
            initial_output3 = model.run(x3)
            if not hasattr(initial_output1, 'numpy'):
                print("It seems that the model is not a tensorflow model...")
                exit()
            print("The model runs...")
        except Exception as e:
            print("The defined model doesn't run!")
            print(e)
            print(traceback.print_exception(e))
        
        if t == 'classification_training':
            training_params = model.c['train_params']
            model.train(
                    tuple(((x1, 2), (x2, 3), (x3, 1))), 
                    10, 
                    1, 
                    0.001,
                    training_params['loss_function'],
                    training_params['optimization_function'],
                    debug=True
                    )
            trained_output1 = model.run(x1)
            trained_output2 = model.run(x2)
            trained_output3 = model.run(x3)

            print("Initial output 1: ", initial_output1)
            print("Initial output 1 class: ", util.get_argmax(initial_output1))
            print("Trained output 1: ", trained_output1)
            print("Trained output 1 class: ", util.get_argmax(trained_output1))
            
            print("Initial output 2: ", initial_output2)
            print("Initial output 2 class: ", util.get_argmax(initial_output2))
            print("Trained output 2: ", trained_output2)
            print("Trained output 2 class: ", util.get_argmax(trained_output2))
            
            print("Initial output 3: ", initial_output3)
            print("Initial output 3 class: ", util.get_argmax(initial_output3))
            print("Trained output 3: ", trained_output3)
            print("Trained output 3 class: ", util.get_argmax(trained_output3))
            
            if util.are_equal(initial_output1, trained_output1):
                print("Something is wrong with the training... Initial and after train outputs are equal so the model is not learning....")
            else:
                print("Training seems to work...")
            
            




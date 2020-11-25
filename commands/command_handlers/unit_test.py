from . import create_args, if_callable_class_function, get_callable_class_functions
#from third_party.tensorflow.test import util
from importlib.util import find_spec
from importlib import import_module
import traceback
from tests.environment_tests import dataset_handler

class UnitTestArgs:
    
    def test_dataset_handler():
        #Defines test function inputs and calls argument parser
        parser_args = {'description':'Unit test for DatasetHandler object'}
        
        add_args = [
            {'name':['command'], 'type':str, 'help':'Main command'},
            {'name':['-test'], 'type':str, 'required':True, 'help':'Name of the command to be tested, use "all" to test all commands'},
            ]
        
        # Parse arguments
        parsed_args = create_args(parser_args, add_args)
        # Check if test is callable
        print(dir(dataset_handler))
        tests = dataset_handler.test
        if parsed_args.test != 'all': 
            if if_callable_class_function(tests, parsed_args.test):
                run(parsed_args.test, parsed_args.testing_data)
            else:
                print("Could not find test ",parsed_args.test,"... \nThese are callable ",get_callable_class_functions(tests))
                exit()

        # Run all tests
        else:
            print("Not yet done")

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
            
            




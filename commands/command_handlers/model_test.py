from . import validate_args, GPU_config, create_args, if_callable_class_function, get_callable_class_functions
from ..command_functions.plot import plot_model
from ..command_functions.test import test_model

class ModelTestArgs:
    
    def data():
        #Defines data information function inputs and calls argument parser
        parser_args = {'description':'Show data information'}
        add_args = [
            {'name':['command'], 'type':str, 'help':'Main command'},
            {'name':['-dh'], 'type':str, 'required':True, 'help':'Name of the dataset handler'},
            {'name':['-ds'], 'type':str, 'required':True, 'help':'Name of the dataset'},
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
            {'name':['-test'], 'type':str, 'required':True, 'help':'Test type'},
            {'name':['-dh'], 'type':str, 'default':None, 'help':'Name of the dataset handler'},
            {'name':['-ds'], 'type':str, 'default':None, 'help':'Name of the dataset'},
            {'name':['-m'], 'type':str, 'default':"NeuralNetworks", 'help':'Model name'},
            {'name':['-c'], 'type':str, 'default':None, 'help':'Name of the configuration file'},
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
            {'name':['-plot'], 'type':str, 'required':True, 'help':'Plot type'},
            {'name':['-dh'], 'type':str, 'default':None, 'help':'Name of the dataset handler'},
            {'name':['-ds'], 'type':str, 'default':None, 'help':'Name of the dataset'},
            {'name':['-m'], 'type':str, 'default':"NeuralNetwork", 'help':'Model name'},
            {'name':['-c'], 'type':str, 'default':None, 'help':'Name of the configuration file'},
            {'name':['--dataset_type'], 'type':str, 'default':'test', 'help':'Dataset type to be used'},
            ]
        
        # Parse arguments
        parsed_args = create_args(parser_args, add_args)
        # Validate input arguments
        validate_args(vars(parsed_args))
        # Run function
        plot_model(vars(parsed_args))



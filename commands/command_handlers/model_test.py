from . import validate_args, GPU_config, create_args, if_callable_class_function, get_callable_class_functions
from ..command_functions.plot import plot_model
from ..command_functions.dataset import dataset_information
from ..command_functions.test import test_model

class ModelTestArgs:
    
    def data():
        #Defines data information function inputs and calls argument parser
        parser_args = {'description':'Show data information'}
        add_args = [
            {'name':['command'], 'type':str, 'help':'Main command'},
            {'name':['-info'], 'type':str, 'required':True, 'help':'Name of the information function'},
            {'name':['-dh'], 'type':str, 'required':True, 'help':'Name of the dataset handler'},
            {'name':['-ds'], 'type':str, 'default':None, 'help':'Name of the dataset'},
            {'name':['-use'], 'type':str, 'default':'train', 'help':'Dataset part used'},
            {'name':['--scale'], 'type':lambda x: (str(x).lower() in ['true', '1', 'yes']), 'default':False, 'help':'True = dataset svaling is applied'},
            {'name':['--balance'], 'type':lambda x: (str(x).lower() in ['true', '1', 'yes']), 'default':True, 'help':'True = dataset balancing is applied'},
            {'name':['--sub_sample'], 'type':int, 'default':None, 'help':'Use a subsample of the dataset'}
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
            {'name':['--scale'], 'type':lambda x: (str(x).lower() in ['true', '1', 'yes']), 'default':True, 'help':'True = dataset scaling is applied'},
            {'name':['--balance'], 'type':lambda x: (str(x).lower() in ['true', '1', 'yes']), 'default':True, 'help':'True = dataset balancing is applied'},
            {'name':['--dataset_type'], 'type':str, 'default':'test', 'help':'Dataset type to be used'},
            {'name':['--store_outputs'], 'type':lambda x: (str(x).lower() in ['true', '1', 'yes']), 'default':False, 'help':'True = dataset scaling is applied'},
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
            {'name':['--scale'], 'type':lambda x: (str(x).lower() in ['true', '1', 'yes']), 'default':True, 'help':'True = dataset scaling is applied'},
            {'name':['--balance'], 'type':lambda x: (str(x).lower() in ['true', '1', 'yes']), 'default':True, 'help':'True = dataset balancing is applied'},
            {'name':['--store_outputs'], 'type':lambda x: (str(x).lower() in ['true', '1', 'yes']), 'default':False, 'help':'True = dataset scaling is applied'},
            {'name':['--sub_sample'], 'type':int, 'default':None, 'help':'Use a subsample of the dataset'},
            {'name':['--plot_dims'], 'type':int, 'default':2, 'help':'Dimensions of the plot'},
            {'name':['--function'], 'type':str, 'default':'PCA', 'help':'Dim reduction function'}
            ]
        
        # Parse arguments
        parsed_args = create_args(parser_args, add_args)
        # Validate input arguments
        validate_args(vars(parsed_args))
        # Run function
        plot_model(parsed_args)



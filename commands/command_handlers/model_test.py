from . import validate_args, GPU_config, create_args, if_callable_class_function, get_callable_class_functions
from ..command_functions.plot import plot_model
from ..command_functions.test import test_model

class ModelTestArgs:
    
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



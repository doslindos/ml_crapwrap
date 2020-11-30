from . import create_args, GPU_config, validate_args

from ..command_functions.dataset import create_dataset, dataset_information
from ..command_functions.train import train_model


class CreateArgs:
    
    def dataset(args=None):
        # args:                 Namespace object containing arguments, if None it is created. (Used mainly for testing)

        if args is None:
            #Defines create function inputs and calls argument parser
            parser_args = {'description':'Create a dataset'}
            add_args = [
                {'name':['command'], 'type':str, 'help':'Main command'},
                {'name':['-dh'], 'type':str, 'required':True, 'help':'Name of the dataset handler, if dataset name (-ds) is not given, this -dh is used in its place'},
                {'name':['-s'], 'type':str, 'default':None, 'help':'Name of the source file. Uses the default value in handler if None is given'},
                {'name':['-ds'], 'type':str, 'default':None, 'help':'Name for the dataset'},
                ]
            # Parse arguments
            parsed_args = create_args(parser_args, add_args)
        else:
            parsed_args = args

        # Validate input arguments
        validate_args(vars(parsed_args))
        # Run function
        create_dataset(parsed_args)

    def train(args=None):
        # args:                 Namespace object containing arguments, if None it is created. (Used mainly for testing)
        
        if args is None:
            #Defines training function inputs and calls argument parser
            parser_args = {'description':'Train a model'}
            add_args = [
                {'name':['command'], 'type':str, 'help':'Main command'},
                {'name':['-dh'], 'type':str, 'required':True, 'help':'Name of the dataset handler'},
                {'name':['-s'], 'type':str, 'default':None, 'help':'Name of the source file'},
                
                {'name':['-ds'], 'type':str, 'default':None, 'help':'Name for the dataset'},
                
                {'name':['-m'], 'type':str, 'default':"NeuralNetworks", 'help':'Model name'},
                {'name':['--batch_size'], 'type':int, 'default':None, 'help':'Batch size, (do not change for sklearn functions)'},
                {'name':['--epochs'], 'type':int, 'default':None, 'help':'Number of times the whole dataset is trained to the model (do not define for sklearn functions)'},
                {'name':['--learning_rate'], 'type':float, 'default':None, 'help':'Learning rate (do not change for sklearn functions)'},
                {'name':['--loss_function'], 'type':str, 'default':None, 'help':'Learning rate (do not change for sklearn functions)'},
                {'name':['--optimization_function'], 'type':str, 'default':None, 'help':'Learning rate (do not change for sklearn functions)'},
                {'name':['--sub_sample'], 'type':int, 'default':None, 'help':'Use a subsample of the dataset'},
                ]
        
            # Parse arguments
            parsed_args = create_args(parser_args, add_args)
        else:
            parsed_args = args

        # Validate input arguments
        validate_args(vars(parsed_args))
        # Set up gpu if you use gpu version of the tensorflow
        #GPU_config()
        # Run function
        train_model(parsed_args)
        

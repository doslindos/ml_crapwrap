from data.setup_functions.data_functions import Preprocess
from data.create_functions import fetch, create
from data import Dataset, data_info
import models
from test_operations import test_functions
from plot_operations import plot_functions, format_data_to_plot
from utils.main_utils import run_function, check_for_func_attr, fetch_model
from tensorflow import config
from pathlib import Path

def GPU_config():
    # For Tensorflow GPU this prevents weird errors in initializing tensorflow
    gpu_devices = config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        config.experimental.set_memory_growth(device, True)

def get_predictions_dict(model_name, datatype):
    # Tries to get a saved file with model output and true label values
    # This file is created so that the model doesn't have to run every time it is tested
    #Select model
    selected_model = models.utils.model_handling_functions.select_weights(model_name)
    return (models.utils.model_handling_functions.read_prediction_file(selected_model, train=datatype), selected_model)

def create_dataset(parsed):
    # Creates a model from data user is defined
    #TODO check id dataset with same name exists!
    
    # Fetch script path from user arguments
    inputs = {'path':parsed.s}
    # Run Fetch Function to retrieve the wanted data
    data, labels = run_function(fetch, parsed.ff, inputs)

    # If the data is not fetched using tensorflow-datasets
    # Creation Function needs to be used for making the dataset
    if parsed.cf is not None:
        if parsed.name is None:
            if '.' in parsed.d:
                name = parsed.d.split('.')
            else:
                name = parsed.d
        else:   
            name = parsed.name
        # Fetch 
        inputs = {'data':data, 'ds_name':name}
        run_function(create, parsed.cf, inputs)

def dataset_information(parsed):
    # Untested

    dataset = Dataset(parsed.d)
    if parsed.merge_key is not None:
        dataset.merge_duplicates(parsed.merge_key)
    # Run the function user is defined and feed inputs
    run_function( data_info, parsed.info, {'dataset':dataset, 'label_key':parsed.l})

def train_model(parsed):
    # Creates a model user is defined

    # Fetch user defined preprocess function from input params
    if parsed.pf is None:
        preprocess_function = parsed.d
    else:
        preprocess_function = parsed.pf
    
    # Initialize Preprocess class
    prep_pipe = Preprocess()
    # Run the preprocessing function
    run_function(
            prep_pipe, 
            preprocess_function, 
            {'dataset_name':parsed.d, 'sub_sample':parsed.sub_sample}
            )
    
    # Retrieve configuration from user input arguments
    # If it is not defined configuration with _basic ending is used
    # TODO
    # Rewrite
    if parsed.c is None:
        conf = parsed.d+'_basic'
    else:
        conf = parsed.c
    
    #Initialize Trainer
    model_handler = models.ModelHandler(prep_pipe.preprocessed_dataset, parsed.m, conf)
    #Run training
    model_handler.train(parsed)

def setup_results(parsed, make_results=True):
    train = parse_dataset_type(parsed.dataset_type)
    results, selected_model = get_predictions_dict(parsed.m, train)   

    if results or not make_results:
        model = (parsed.m, selected_model)
        if not results:
            results = None
    else:
        print("Creating predictions file...")
        
        #Preprocess dataset
        if parsed.pf is None:
            preprocess_function = parsed.d
        else:
            preprocess_function = parsed.pf
        
        prep_pipe = Preprocess()
        run_function(
                prep_pipe, 
                preprocess_function, 
                {'dataset_name':parsed.d, 'sub_sample':parsed.sub_sample})

        model_handler = models.ModelHandler(prep_pipe.preprocessed_dataset, parsed.m, selected_model)
        
        #Run test
        results = model_handler.test(parsed.test, None, train=train)
        model = model_handler.model
        
    return (results, model)

def test_model(parsed):
    if 'gui' in parsed.test:
        make_results = False
    else:
        make_results = True

    results, model = setup_results(parsed, make_results)
    if make_results and not results:
        exit()
    
    if parsed.pf is None:
        pf = parsed.d
    else:
        pf = parsed.pf
    
    if isinstance(model, Path):
        print(model)
        exit()
    
    inputs = {'results':results, 'model':model}
    if check_for_func_attr(getattr(test_functions, parsed.test), 'preprocess_function'):
        inputs['preprocess_function'] = pf

    # Run the function user is defined and feed inputs
    run_function(test_functions, parsed.test, inputs)

def plot_model(parsed):
    # Steps to run a plot with model output

    # Set up results and model
    results, model = setup_results(parsed)
    if not results:
        exit()
    
    # Format the data to be plotted and assing results to a dict
    labels, data = format_data_to_plot(results, parsed.plot_dims, parsed.function)
    inputs = {'labels':labels, 'data':data}
    
    # Run the function user is defined and feed inputs
    run_function(plot_functions, parsed.plot, inputs)

def error(msg):
    # Not sure does this work...
    print(msg)
    exit()

def parse_dataset_type(datatype):
    # Used to check for dataset used, train or test
    if datatype == 'train':
        train = True
    else:
        train = False

    return train

def validate_args(args):
    # Validates arguments
    # TODO Rewrite this
    function_call_args = {'test':test_functions, 'plot':plot_functions, 'info':data_info}
    for name, func in function_call_args.items():
        if name in args:
            if args[name] not in dir(func):
                error(args[name]+ " is not a valid function name for -"+ name+ " in "+ args['command'])

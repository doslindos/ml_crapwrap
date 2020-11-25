from tests.model_tests import test_functions
from plotting import plot_functions
from data import data_info
from . import config, run_function, select_weights, read_prediction_file, ModelHandler, load_data

def GPU_config():
    # For Tensorflow GPU this prevents weird errors in initializing tensorflow
    gpu_devices = config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        config.experimental.set_memory_growth(device, True)

def error(msg):
    # Not sure does this work...
    print(msg)
    exit()

def validate_args(args):
    # Validates arguments
    # TODO Rewrite this
    function_call_args = {'test':test_functions, 'plot':plot_functions, 'info':data_info}
    for name, func in function_call_args.items():
        if name in args:
            if args[name] not in dir(func):
                print(args[name]+ " is not a valid function name for -"+ name+ " in "+ args['command'])
                exit()

def get_predictions_dict(model_name, datatype):
    # Tries to get a saved file with model output and true label values
    # This file is created so that the model doesn't have to run every time it is tested
    #Select model
    selected_model = select_weights(model_name)
    return (read_prediction_file(selected_model, train=datatype), selected_model)

def setup_results(parsed, make_results=True):
    dset = parse_dataset_type(parsed.dataset_type)
    results, selected_model = get_predictions_dict(parsed.m, dset)   

    if results or not make_results:
        model = (parsed.m, selected_model)
        if not results:
            results = None
    else:
        print("Creating predictions file...")
        
        # Load data
        dataset = load_data(parsed.d)

        train, validation, test = dataset.fetch_preprocessed_data(parsed.sub_sample)
        
        model_handler = ModelHandler((train, validation, test), parsed.m, selected_model)
        
        #Run test
        results = model_handler.test(parsed.test, None, train=dset)
        model = model_handler.model
        
    return (results, model)

def parse_dataset_type(datatype):
    # Used to check for dataset used, train or test
    if datatype == 'train':
        train = True
    else:
        train = False

    return train


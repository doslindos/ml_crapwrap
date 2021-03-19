from tests.model_tests import test_functions
from plotting import plot_functions
from data import data_info
from . import Path, config, run_function, select_weights, read_prediction_file, ModelHandler, load_data, open_dirGUI

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

def get_predictions_dict(model_name, fname):
    # Tries to get a saved file with model output and true label values
    # This file is created so that the model doesn't have to run every time it is tested
    #Select model
    selected_model = select_weights(model_name)
    return (read_prediction_file(selected_model, prediction_filename=fname), selected_model)

def setup_results(parsed):
    fname = pred_filename_generator(parsed)
    results, selected_model = get_predictions_dict(parsed.m, fname)   

    if results:
        model = (parsed.m, selected_model)
        if not results:
            results = None
    else:
        if selected_model is not None:
            if parsed.dh is None:
                # Select the data
                path = open_dirGUI(Path("data", "handlers"))
                handler = path.parent.parent
                ds = path.name
                source = None
            else:
                # Use dataset which is given
                handler = parsed.dh
                ds = parsed.ds
                source = None
        else:
            print("You did not select a model...")
            exit()

        print("Setting up the dataset...")
        # Load data
        dataset = load_data(ds, source, handler)

        train, validation, test = dataset.fetch_preprocessed_data(parsed.sub_sample, parsed.scale, parsed.balance)
        
        print("Setting up the model...")
        model_handler = ModelHandler((train, validation, test), parsed.m, selected_model)
        
        print("Running test...")
        if not parsed.store_outputs:
            # Don't store outputs/predictions
            fname = None

        #Run test
        if 'test' in vars(parsed).keys():
            test = parsed.test
        else:
            test = None

        results = model_handler.test(test_name=test, results=None, fname=fname, dstype=parsed.dataset_type)
        model = model_handler.model
        
        if test is not None:
            exit()
        
    return (results, model)

def pred_filename_generator(args):
    # Used to check for dataset used
    dstype = args.dataset_type
    ds = args.ds
    if ds is None:
        ds = args.dh

    return ds+"_"+dstype+'_label_output.pkl'

def setup_ui(parsed):
    selected_model = select_weights(parsed.m)
    if parsed.m is None:
        # Search for the model name (after folder named models)
        model_name = [ f.name for f in list(selected_model.parents) if str(f.parent.name) == 'models']
        # In case there are 2 folders named models in path use the last one
        # TODO make naming a model = models prohibited
        model_name = model_name[-1]
    else:
        model_name = parsed.m

    model_handler = ModelHandler(None, model_name, selected_model)

    return model_handler.model

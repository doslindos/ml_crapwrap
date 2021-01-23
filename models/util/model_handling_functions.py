from .. import npsave, npload, nparray, npappend, npexpand, datetime, Path, jsondump, jsonload, signature, open_dirGUI, getcwd, argmax, get_module, pkldump, pklload
from joblib import dump as joblibdump, load as joblibload

def create_folder(path):
    # Creates an folder if it doesn't exist
    # In:
    #   path:                       Path object, path to where folder is searched and created
    
    if not path.exists():
        path.mkdir()
    
def save_configuration(configuration, model_name, path):
    # Save configurations
    # In: 
    #   configuration:              dict, all models parameters
    #   model_name:                 str, name of the model
    #   path:                       Path object, path to the model
    #print(configuration, model_name)   
    with path.joinpath('config.json').open('w') as f:
        jsondump({model_name:configuration}, f)
    
def load_configuration(path):
    # Load configurations
    # In:
    #   path:                       Path object, path to the configurations file
    # Out:
    #   (model_name, config):       tuple, (str, dict), successfully fetched model_name and configuration dict
    
    if path.joinpath('config.json').exists():
        with path.joinpath('config.json').open('r') as f:
            conf = jsonload(f)
    else:
        print(path.name, " is not a saved model")
        exit()
    model_name, config = next(iter(conf.items()))

    return (model_name, config)

def save_weights(weights, biases, path):
    # Save model weights
    # In:
    #   weights:                    list, [numpy arrays]
    #   biases:                     list, [numpy arrays]
    #   path:                       Path object
    # Out:
    #   savepath:                   Path object

    #Timestamp
    now = datetime.now()
    
    #Create weights folder in <dirname> directory
    create_folder(path.parent.parent)
    create_folder(path.parent)
    create_folder(path)

    #Create a directory (name = timestamp) in the weights directory if not exists
    path = path / now.strftime("%d_%m_%Y_%H-%M")
    create_folder(path)
    print(path)

    #Save weights in currently created directory
    with path.joinpath('weights.npy').open('wb') as f:
        npsave(f, nparray(weights))
    
    if biases is not None:
        with path.joinpath('bias.npy').open('wb') as f:
            npsave(f, nparray(biases))

    return path

def load_weights(path):
    # Load weights into a model
    # In;
    #   path:                       Path object
    # Out:
    #   (weights, bias):            tuple, (numpy array, numpy array)

    with path.joinpath('weights.npy').open('rb') as f:
        weights = npload(f, allow_pickle=True)

    if path.joinpath('bias.npy').exists():
        with path.joinpath('bias.npy').open('rb') as f:
            bias = npload(f, allow_pickle=True)
    else:
        bias = None
    
    return (weights, bias)

def save_sk_model(model, path):
    # Save scikit models
    # In:
    #   model:                      Sklearn object
    #   path:                       Path object
    # Out:
    #   savepath:                   Path object

    #Timestamp
    now = datetime.now()
    
    #Create weights folder in <dirname> directory
    create_folder(path.parent)
    create_folder(path)

    #Create a directory (name = timestamp) in the weights directory if not exists
    path = path / now.strftime("%d_%m_%Y_%H-%M")
    create_folder(path)
    #print(path)

    #Save weights in currently created directory
    
    joblibdump(model, path.joinpath('model.joblib'))
    
    return path

def load_sk_model(path):
    # Load weights into a model
    # In;
    #   path:                       Path object
    # Out:
    #   model:                      Sklearn object

    #print(path)
    return joblibload(path.joinpath('model.joblib'))

def select_weights(model_name):
    # Select weights to load
    # In:
    #   model_name:                 str
    # Out:
    #   <path to selected model>:   str
    
    path = Path(getcwd()).joinpath('models/'+model_name+'/saved_models/')
    if not path.exists():
        print("No saved models!")
        exit()
    
    return open_dirGUI(path)    

def handle_init(model, path, confs):
    # Handle model initialization
    # In:
    #   model:                      Model object
    #   path:                       Path where saved models are searched, str when model is created
   
    def load(load_path):
        if load_path.is_dir():
            #Load configurations and apply them
            conf_name, configurations = load_configuration(path)
            #print(conf_name, configurations)
            model.conf_name = conf_name
            model.c = configurations
            
            #Load model
            model.load_path = load_path
            model.load(load_path)
        else:
            print("Path is not a dircetory...")
            exit()
    
    if isinstance(path, Path):
        # Handle initalization with path as string
        # This should be invoked when creating a new model (train)
        if path.is_file():
            c = get_module(path)
            model.conf_class_name = path.parts[-3:-1]
            model.conf_name = path.stem
            model.load_path = None
            model.c = c.conf
        # If path is load user is shown weight selection interface
        elif path.is_dir():
            load(path)
        # If path is not a string it is assumed to be a path for the weights
        else:
            path = Path(select_weights(path))
            load(path)

def read_prediction_file(path, prediction_filename='label_output.pkl', train=False):
    # Reads prediction file
    # In:
    #   path:                       Path object, path where to save predictions file
    #   prediction_filename:        str
    #   train:                      bool, train or test dataset
    # Out:
    #   results:                    dict, keys = labels and values = model outputs
    
    
    if not path.joinpath(prediction_filename).exists():
        return False
    else:
        #Read label model output file
        with path.joinpath(prediction_filename).open('rb') as fl:
            results = pklload(fl)
        
        return results

def map_params(function, commandline_params, configuration={}):
    # Map params to function

    func_params = {}
    for param in signature(function).parameters:
        if param in commandline_params:
            if commandline_params[param] is not None:
                func_params[param] = commandline_params[param]
                if 'train_params' in configuration.keys():
                    if param in configuration['train_params'].keys():
                        configuration['train_params'][param] = commandline_params[param]
            elif 'train_params' in configuration.keys():
                func_params[param] = configuration['train_params'][param]
            else:
                print("You have to specify parameter ", param, "!")
    return func_params

def create_prediction_file(path, dataset, model, prediction_filename=None):
    # Takes model output - label pairs and stores them in a pickle file
    # In:
    #   path:                       Path object, path where to save predictions file
    #   dataset:                    Tensorflow dataset object
    #   model:                      Model object
    #   prediction_filename:        str
    # Out:
    #   results:                    dict, keys = labels and values = model outputs

    print("Creating a predictions file...")
    results = {}

    for batch in dataset.batch(dataset.cardinality()):
        x = batch[0]
        y = batch[1]
        out = model.run(**map_params(model.run, {'x':x, 'training':False}))
        
        # Handle output as a tensor
        if hasattr(out, 'numpy'):
            out = out.numpy()
        if hasattr(y, 'numpy'):
            y = y.numpy()
        
        for i, label in enumerate(y):
            
            # If output values are not numpy arrays
            if len(out.shape) > 1:
                instance = out[i].reshape((1, out[i].shape[-1]))
            else:
                instance = nparray(out[i]).reshape(1, 1)

            if label not in results.keys():
                results[label] = instance
            else:
                #Stack outputs
                results[label] = npappend(results[label], instance, axis=0)
    
    if prediction_filename is not None:
        with path.joinpath(prediction_filename).open('wb') as fs:
            pkldump(results, fs)
    
    return results


from importlib import import_module
from importlib.util import find_spec
from inspect import signature
from csv import reader as csv_reader
from pathlib import Path

from numpy import append as npappend, expand_dims as expdims, array as nparray
from tensorflow import data, squeeze as tfsqueeze, stack as tfstack
import cv2

def get_xy(instance):
    if isinstance(instance, dict):
        #print(instance)
        x = instance['x']
        y = instance['y']
    elif isinstance(instance, tuple) or isinstance(instance, list):
        x = instance[0]
        y = instance[1]
    else:
        print("Data instance handling for ", type(instance), " not made...")

    return (x, y)

def get_callable_class_functions(class_obj):
    # Takes a class object and returns a list of its callable functions
    # In:
    #   class_obj:                      class object
    # Out:
    #   function list:                  list of callable function names

    return [func for func in dir(class_obj) if not func.startswith('__') and callable(getattr(class_obj, func))]


def if_callable_class_function(class_obj, function):
    # Takes class and function name
    # and checks if the function is in it
    # In:
    #   class_obj:                      class object used
    #   function:                       str, function name
    # Out:
    #   boolean:                        function is found in classes callable functions or not

    callable_functions = get_callable_class_functions(class_obj)
    print(callable_functions)
    if function in callable_functions:
        return True
    else:
        return False

def take_image_screen(size=[]):
    pass

def import_with_string(string):
    if find_spec(string) is not None:
        return import_module(string)
    else:
        print("Module: ", string, " was not found")
        exit()

def fetch_model(model_name, conf_name):
    # Fetch the model
    # In:
    #   path:                       str, path to the model
    
    model_module = import_with_string("models."+model_name+".model")
    #import_module('models.'+model_name+'.model')
    return model_module.Model(conf_name)

def fetch_resource(path, desired_shape=None, desired_dtype=None):
    # Handle resource fetching
    # In:
    #   path:                       Path Object

    if path.exists():
        suf = path.suffix
        
        # Handle unspecfied data type
        if desired_dtype is not None:
            dtype = desired_dtype
        else:
            dtype = 'float32'

        if suf in ['.jpg', '.png']:
            if desired_shape is None:
                # No desired input shape defined
                # Return image as numpy array, with RGB color type
                return nparray(cv2.imread(path.as_posix(), cv2.COLOR_BGR2RGB), dtype=dtype)
            else:
                # Define shapes
                color_dim = desired_shape[-1]
                h = desired_shape[0]
                w = desired_shape[1]
                
                # Handle different color types
                if color_dim == 1:
                    col = cv2.COLOR_BGR2GRAY
                elif color_dim == 3:
                    col = cv2.COLOR_BGR2RGB

                # Read image
                img = cv2.cvtColor(cv2.imread(path.as_posix()), col)
                # Resize image for the model
                resized = cv2.resize(img, (h, w))
                
                # Reshape it for the model (add batch dimension)
                resized = resized.reshape((1, h, w, color_dim))
                # Convert to a specific data type
                resized = resized.astype(dtype)
                
                return (img, resized)

        elif suf in ['.csv']:
            
            def error(value):
                print("Input must be numeric...  your input ", value ," can't be converted to a float...")
                exit()
            
            # Desired input array length
            array_len = desired_shape[-1]

            # Read csv file
            csv_file = csv_reader(path.open('r', encoding='utf8'), delimiter=';')
            # Loop through rows
            n_rows = []
            for row_num, row in enumerate(csv_file):
                # Loop through instances in a row
                for i, value in enumerate(row):
                    # Convert instance to float and replace the old value with converted
                    if isinstance(value, str) or isinstance(value, int):
                        try:
                            value = float(value)
                        except ValueError:
                            error(value)
                    else:
                        error(value)

                    row[i] = value
                
                n_rows.append(row)

            n_rows = nparray(n_rows, dtype=dtype)
            
            if desired_shape is not None:
                if len(n_rows) > 1:
                    n_rows = n_rows.reshape((1, len(n_rows), array_len))
                else:
                    n_rows = n_rows[0].reshape((1, array_len))


            return (n_rows, n_rows)


def dataset_generator(dataset, batch_size):
    # get instances out of a dataset
    # In:
    #   dataset:                    tensorflow dataset object
    #   num_instances:              number of instances taken
    if hasattr(dataset, 'batch'):
        dataset = dataset.batch(batch_size)

    for inst in dataset:
        yield get_xy(inst)

def get_dataset_info(datasets, length=False):
        
    def get_info(ds):
        ds_length = None
        if isinstance(ds, dict):
            # Handle custom created datasets
            data_size = ds['x'][0].shape
            data_type = ds['x'][0].dtype
            if length:
                ds_length = len(ds)
        else:
            # Handle tfds datasets
            instance = [i for i in ds.take(1)][0][0]
            data_size = instance.shape
            data_type = instance.dtype
            #doesn't work
            #ds_length = data.experimental.cardinality(dataset).numpy()
            #SLOW
            if length:
                ds_length = sum(1 for _ in ds)
        
        return (ds_length, data_size, data_type)

    if isinstance(datasets, tuple):
        ds_info = []
        for dataset in datasets:
            ds_info.append(get_info(dataset))
    else:
        ds_info = get_info(datasets)
    
    return ds_info

def results_to_nplist(results):
    if isinstance(results, dict):
        labels = []
        initial = True
        for i, (label, result) in enumerate(results.items()):
            #print(label, result.shape)
            
            for data_instance in result:
                data_instance = expdims(data_instance, axis=0)
                if initial:
                    data = data_instance
                    initial = False
                else:
                    data = npappend(data, data_instance, axis=0)
                labels.append(label)

        #print(data.shape)
        return (data, labels)
    else:
        print("Results is not a dict...")

def run_function(module, func_name, inputs):
    # Runs a function
    # In:
    #   module:                 python module
    #   func_name:              str, name of the function in the module
    #   inputs:                 dict, input dictionary
    if func_name in dir(module):
        results = getattr(module, func_name)(**inputs)
    else:
        print("Module ",module," has no function: ", func_name)
        exit()

    if results is not None:
        return results

def check_for_func_attr(function, attribute_name):
    if attribute_name in list(signature(function).parameters.keys()):
        return True
    else:
        return False

def get_function_attr_values(function, attrs=None):
    # Fetch default values from function attributes
    # In:
    #   function:                   function module, function where values are fetched
    #   attrs:                      None returns all, str returns only the one specified or list returns all listed attribute values
    # Out:
    #   dict:                       Contains specified attribute as key and its default value as value

    def check_attr(attr):
        if attr in list(func_attrs.parameters.keys()):
            return func_attrs.parameters[attr].default
        else:
            print("Attr ", attr, " not found...")
    
    func_attrs = signature(function)
    if attrs is not None:
        if isinstance(attrs, list):
            return_attrs = {}
            for attr in attrs:
                at = check_attr(attr)
                return_attrs[attr] = at
            return return_attrs
        elif isinstance(attrs, str):
            return {attrs:check_attr(attrs)}
    else:
        return {k:v.default for k, v in func_attrs.parameters.items()}


def duplicate_along(data, along_ax, label=None):
    data = tfsqueeze(tfstack([data for i in range(3)], axis=along_ax))
    return (data, label)

def list_files_in_folder(folder_path, suffix='*.py'):
    files = Path(folder_path).rglob(suffix)
    return [f.name.split('.')[0] for f in files]

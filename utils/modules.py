from importlib import import_module
from importlib.util import find_spec
from . import Path
from GUI import open_fileGUI
from utils.utils import recursive_file_search

def get_module(path_to_module):
    # Takes the path to the module wanted to fetch and returns the module
    # In:
    #   path_to_module:                 str, module name as a string
    # Out:
    #   fetched modul:                  module
    from importlib.util import spec_from_file_location, module_from_spec
    if isinstance(path_to_module, str):
        if find_spec(path_to_module) is not None:
            return import_module(path_to_module)
        else:
            print("Module: ", path_to_module, " was not found")
            exit()
    elif isinstance(path_to_module, Path):
        if path_to_module.exists():
            spec = spec_from_file_location(path_to_module.name, path_to_module)
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            print("No module in ", path_to_module)

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
    if function in callable_functions:
        return True
    else:
        return False

def search_conf(model, conf):
    # Makes a recursive search for a configurations file from configurations files of a given model
    # Opens a GUI if a file is not found with given name or there are several
    # In:
    #   model:                      str, name of the model
    #   conf:                       str, name of the configurations file
    # Out:
    #   Path object:                path to the configuration file
    
    # Path from which to search configurations
    model_confs_path = Path('models', model, 'configurations')

    # If None is given as conf open the GUI
    if conf is None:
        # Open a gui to choose conf file
        return open_fileGUI(model_confs_path)
    else:
        # Search the conf file
        # If searched file is given with suffix
        if '.' in conf:
            full = True
        else:
            full = False
        
        # Make the recursive search
        files = recursive_file_search(model_confs_path, '*.*', full)
        # If only one file matches the name given
        files_with_given_name = list(files.keys()).count(conf)
        if files_with_given_name == 1:
            return files[conf]
        else:
            print("More or less than one file called ", conf ," found...")
            return open_fileGUI(model_confs_path)

def fetch_model(model_name, conf_name):
    # Fetch the model
    # In:
    #   model_name:                     str, name of the model
    #   conf_name:                      str, name of the model configration file
    
    model_module = get_module("models."+model_name+".model")
    if isinstance(conf_name, str) or conf_name is None:
        conf_name = search_conf(model_name, conf_name)

    return model_module.Model(conf_name)

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from GUI import open_fileGUI

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

def fetch_model(model_name, conf_name):
    # Fetch the model
    # In:
    #   model_name:                     str, name of the model
    #   conf_name:                      str, name of the model configration file
    
    model_module = get_module("models."+model_name+".model")
    if isinstance(conf_name, str):
        conf_name = open_fileGUI(
                Path("models", model_name,"configurations"), 
                (('python files', '*.py'), )
                )
    return model_module.Model(conf_name)

from importlib import import_module
from importlib.util import find_spec

def get_module(path_to_module):
    # Takes the path to the module wanted to fetch and returns the module
    # In:
    #   path_to_module:                 str, module name as a string
    # Out:
    #   fetched modul:                  module

    if find_spec(path_to_module) is not None:
        return import_module(path_to_module)
    else:
        print("Module: ", path_to_module, " was not found")
        exit()

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
    
    return model_module.Model(conf_name)

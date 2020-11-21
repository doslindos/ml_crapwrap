
from inspect import signature

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
        # If attributes searched is None return all attributes of defined function as a dict
        return {k:v.default for k, v in func_attrs.parameters.items()}

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

from utils.modules import get_module

class Model:

    def __init__(self, model_name):
        # Retrieve model script
        tst = get_module(model_name)
        # Retrieve all non built in functions and variables from model script
        # set them to be callable via Model = self
        functions = [x for x in dir(tst) if not x.startswith('__')]
        for function in functions:
            print(function)
            setattr(self, function, getattr(tst, function))
    

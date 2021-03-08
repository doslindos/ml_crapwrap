from . import Path, run_function, if_callable_class_function, test_functions
from .utils import setup_results, setup_ui

def test_model(parsed):
    
    results, model = setup_results(parsed)
    if parsed.store_outputs and not results:
        exit()
    
    if isinstance(model, Path):
        print(model)
        exit()
    
    inputs = {'results':results, 'model':model}
    if if_callable_class_function(test_functions, parsed.test):
        # Run the function user is defined and feed inputs
        run_function(test_functions, parsed.test, inputs)
    else:
        print("Test ", parsed.test, " not found in tests/model_tests/test_functions.py")

def UI(parsed):

    model = setup_ui(parsed)
    print(model)

    if parsed.ui == 'gui':
        test_functions.testing_gui(model)
    if parsed.ui == 'cli':
        test_functions.testing_cli(model)

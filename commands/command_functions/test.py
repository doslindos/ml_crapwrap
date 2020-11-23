from . import Path, run_function, if_callable_class_function, test_functions
from .utils import setup_results

def test_model(parsed):
    if 'gui' in parsed.test:
        make_results = False
    else:
        make_results = True

    results, model = setup_results(parsed, make_results)
    if make_results and not results:
        exit()
    
    if parsed.pf is None:
        pf = parsed.d
    else:
        pf = parsed.pf
    
    if isinstance(model, Path):
        print(model)
        exit()
    
    inputs = {'results':results, 'model':model}
    if if_callable_class_function(test_functions, parsed.test):
        # Run the function user is defined and feed inputs
        run_function(test_functions, parsed.test, inputs)
    else:
        print("Test ", parsed.test, " not found in tests/model_tests/test_functions.py")


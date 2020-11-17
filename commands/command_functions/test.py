from . import Path, run_function, check_for_func_attr
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
    if check_for_func_attr(getattr(test_functions, parsed.test), 'preprocess_function'):
        inputs['preprocess_function'] = pf

    # Run the function user is defined and feed inputs
    run_function(test_functions, parsed.test, inputs)


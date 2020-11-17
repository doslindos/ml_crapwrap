from . import run_function
from .utils import setup_results
from plotting import plot_functions, format_data_to_plot

def plot_model(parsed):
    # Steps to run a plot with model output

    # Set up results and model
    results, model = setup_results(parsed)
    if not results:
        exit()
    
    # Format the data to be plotted and assing results to a dict
    labels, data = format_data_to_plot(results, parsed.plot_dims, parsed.function)
    inputs = {'labels':labels, 'data':data}
    
    # Run the function user is defined and feed inputs
    run_function(plot_functions, parsed.plot, inputs)



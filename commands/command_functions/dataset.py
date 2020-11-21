from . import run_function, Dataset
from data.create_functions import fetch, create

def load_data(name):
    # Initialize Dataset
    dataset = Dataset(parsed.d)
    # Load the data
    dataset.load()
    return dataset

def create_dataset(parsed):
    load_data(parsed.d)

def dataset_information(parsed):
    # Untested
    
    dataset = load_data(parsed.d)

 #   if parsed.merge_key is not None:
 #       dataset.merge_duplicates(parsed.merge_key)
    # Run the function user is defined and feed inputs
    run_function( data_info, parsed.info, {'dataset':dataset, 'label_key':parsed.l})


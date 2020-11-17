from . import run_function, Dataset
from data.create_functions import fetch, create

def create_dataset(parsed):
    # Creates a model from data user is defined
    #TODO check id dataset with same name exists!
    
    # Fetch script path from user arguments
    inputs = {'path':parsed.s}
    # Run Fetch Function to retrieve the wanted data
    data, labels = run_function(fetch, parsed.ff, inputs)

    # If the data is not fetched using tensorflow-datasets
    # Creation Function needs to be used for making the dataset
    if parsed.cf is not None:
        if parsed.name is None:
            if '.' in parsed.d:
                name = parsed.d.split('.')
            else:
                name = parsed.d
        else:   
            name = parsed.name
        # Fetch 
        inputs = {'data':data, 'ds_name':name}
        run_function(create, parsed.cf, inputs)

def dataset_information(parsed):
    # Untested

    dataset = Dataset(parsed.d)
    if parsed.merge_key is not None:
        dataset.merge_duplicates(parsed.merge_key)
    # Run the function user is defined and feed inputs
    run_function( data_info, parsed.info, {'dataset':dataset, 'label_key':parsed.l})


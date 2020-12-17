from . import run_function, load_data, if_callable_class_function
from data import data_info

def create_dataset(parsed):
    load_data(parsed.ds, parsed.s, parsed.dh)

def dataset_information(parsed):

    ds_handler = load_data(parsed.ds, None, parsed.dh)
    data = ds_handler.fetch_preprocessed_data(parsed.sub_sample, parsed.scale, parsed.balance)
 
    # Choose dataset to be used
    if parsed.use == 'train':
        dataset = data[0]
    elif parsed.use == 'validate':
        dataset = data[1]
    elif parsed.use == 'test':
        dataset = data[2]
    else:
        print(parsed.use, " not found...")
        print("train, validate and test are available inputs for -use")
        exit()
    
    # Run the function user is defined and feed inputs
    if if_callable_class_function(data_info, parsed.info):
        run_function( 
                data_info, 
                parsed.info, 
                {'dataset':dataset})
    else:
        print("Function ", parsed.info, " not found in data/data_info.py")

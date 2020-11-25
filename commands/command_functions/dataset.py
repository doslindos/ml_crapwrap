from . import run_function, load_data

def create_dataset(parsed):
    load_data(parsed.d)

def dataset_information(parsed):
    # Untested
    
    dataset = load_data(parsed.d)
    data = dataset.fetch_raw_data(parsed.sub_sample)

 #   if parsed.merge_key is not None:
 #       dataset.merge_duplicates(parsed.merge_key)
    # Run the function user is defined and feed inputs
    if if_callable_class_function(data_info, parsed.info):
        run_function( data_info, parsed.info, {'dataset':dataset, 'label_key':parsed.l})
    else:
        print("Function ", parsed.info, " not found in data/data_info.py")

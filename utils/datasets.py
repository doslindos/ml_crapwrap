
#TODO write comments

def get_xy(instance):
    if isinstance(instance, dict):
        #print(instance)
        x = instance['x']
        y = instance['y']
    elif isinstance(instance, tuple) or isinstance(instance, list):
        x = instance[0]
        y = instance[1]
    else:
        print("Data instance handling for ", type(instance), " not made...")

    return (x, y)

def dataset_generator(dataset, batch_size):
    # get instances out of a dataset
    # In:
    #   dataset:                    tensorflow dataset object
    #   num_instances:              number of instances taken
    # Out:
    #   a batch of the dataset

    if hasattr(dataset, 'batch'):
        dataset = dataset.batch(batch_size)

    for inst in dataset:
        yield get_xy(inst)

def get_dataset_info(datasets, length=False):
        
    def get_info(ds):
        ds_length = None
        if isinstance(ds, dict):
            # Handle custom created datasets
            data_size = ds['x'][0].shape
            data_type = ds['x'][0].dtype
            if length:
                ds_length = len(ds)
        else:
            # Handle tfds datasets
            instance = [i for i in ds.take(1)][0][0]
            data_size = instance.shape
            data_type = instance.dtype
            if length:
                ds_length = sum(1 for _ in ds)
        
        return (ds_length, data_size, data_type)

    if isinstance(datasets, tuple):
        ds_info = []
        for dataset in datasets:
            ds_info.append(get_info(dataset))
    else:
        ds_info = get_info(datasets)
    
    return ds_info

from . import Path, jsondump, jsonload, Counter, itemgetter, import_module

def import_error(mod, ds_name, path):
    print("You do not have a ",mod," for ", ds_name, " in ", path)
    print("If you want to create a dataset ", ds_name, " check out README in data folder...")
    exit()

class DatasetHandler:
    # Calls handlers

    def __init__(self, handler_name, dataset_name, source):
        # Initial variables and dataset fetch
        # In: 
        #   handler_name:               str, name of the handler to use.
        #   dataset_name:               str, name for/of the dataset.
        #   source:                     str, input for the fetcher .
        
        # Use handlers name as dataset name if it is not given
        if dataset_name is None:
            dataset_name = handler_name
        
        # Make a list from the parameters
        params = [handler_name, dataset_name]
        if source is not None:
            # Add source to the params only if it is not None
            params.append(source)

        # Define paths to search a fetcher and preprocesser
        handler_path = Path("data", "handlers")

        # Create DataPreprocessor
        p_path = handler_path.joinpath(handler_name, "preprocess.py")
        if p_path.exists():
            processor = import_module("data.handlers."+handler_name+".preprocess")
            self.data_preprocessor = processor.DataPreprocessor(*params)
        else:
            import_error("DataPreprocessor", handler_name, p_path)
        
    def load(self, sub_sample=None):
        # Actual fetching of the data
        self.data_preprocessor.load_data(sub_sample)
        
    def fetch_raw_data(self, sub_sample=None):
        # Fetches the unpreprocessed data
        return self.data_preprocessor.get_data(sub_sample)

    def fetch_preprocessed_data(self, sub_sample=None, scale=True, balance=True, new_split=False):
        dataset = self.fetch_raw_data(sub_sample)
        return self.data_preprocessor.preprocess(
                                    dataset, 
                                    scale, 
                                    balance, 
                                    new_split
                                    )


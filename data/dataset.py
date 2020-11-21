from . import Path, jsondump, jsonload, Counter, itemgetter, import_module

def import_error(mod, ds_name, path):
    print("You do not have a ",mod," for ", ds_name, " in ", path)
    print("If you want to create a dataset ", ds_name, " check out README in data folder...")
    exit()

class Dataset:
    # Handles datasets

    def __init__(self, dataset_name):
        # Initial variables and dataset fetch
        # In: 
        #   dataset_name:               str, name of the dataset to be used
        
        # Define paths to search a fetcher and preprocesser
        handler_path = Path("data", "handlers")

        # Create DataFetcher
        if handler_path.joinpath(dataset_name, "fetch.py").exists():
            # Convert to posix to use import module
            fetcher = import_module("data.handlers."+dataset_name+".fetch")
            self.data_fetcher = fetcher.DataFetcher(dataset_name)
        else:
            import_error("DataFetcher", dataset_name, fetcher_path)

        # Create DataPreprocessor
        if handler_path.joinpath(dataset_name, "preprocess.py").exists():
            processor = import_module("data.handlers."+dataset_name+".preprocess")
            self.data_preprocessor = processor.DataPreprocessor()
        else:
            import_error("DataPreprocessor", dataset_name, preprocessor_path)
        
    def load(self):
        # Actual fetching of the data
        self.data_fetcher.load_data()
        
    def fetch_raw_data(self, sub_sample=None):
        return self.data_fetcher.get_data(sub_sample)

    def fetch_preprocessed_data(self, sub_sample=None):
        dataset = self.fetch_raw_data(sub_sample)
        self.data_preprocessor.preprocess(dataset)

    def label_counts(self, key):
        #
        if hasattr(self, 'data_lists'):
            print(type(self.data_lists))
            data = self.data_lists[key]
            d_types = set([type(d) for d in data])
        
            for i, d in enumerate(data):
                if len(d_types) > 1:
                    if isinstance(d, list):
                        data[i] = max(d)
                else:
                    if isinstance(d, list):
                        data[i] = tuple(d)
        
            #print("Number of different values in ", key, len(set(data)))
            #print(set(data))
            data = dict(sorted(Counter(data).items(), key=lambda kv: kv[0]))
            return {str(key):value for key, value in data.items()}
        else:
            print("Your dataset seems to be prebuild one... This function doesn't work with prebuild datasets")

    def merge_duplicates(self, based_on_key):
        if hasattr(self, 'data_lists'):
            data = self.data_lists[based_on_key]
            duplicate_data = {}
            for i, d in enumerate(data):
                if tuple(d) not in duplicate_data.keys():
                    duplicate_data[tuple(d)] = [i]
                else:
                    duplicate_data[tuple(d)].append(i)
        
            delete_indexes = []
            for l, (key, data_list) in enumerate(self.data_lists.items()):
                for dupli_value, indexes in duplicate_data.items():
                    if len(indexes) > 1:
                        for i, index in enumerate(indexes):
                            if i == 0:
                                merged = [data_list[index]]
                                merge_index = index
                            else:
                                if data_list[index] not in merged:
                                    merged.append(data_list[index])
                                if l == 0:
                                    delete_indexes.append(index)

                        if len(merged) == 1:
                            merged = merged[0]

                        data_list[merge_index] = merged
            
                for index in reversed(sorted(delete_indexes)):
                    del data_list[index]

        else:
            print("Your dataset seems to be prebuild one... This function doesn't work with prebuild datasets")

    def get_data(self, sub_sample=None):

        def fetch_from_json():
            dataset = jsonload(path.open('r', encoding='utf-8'))
            if sub_sample is not None:
                dataset = dataset[:sub_sample]

            data_lists = {}
            for sample in dataset:
                for key, value in sample.items():
                    if hasattr(self, key):
                        getattr(self, key).append(value)
                    else:
                        setattr(self, key, [])
                        self.data_key_list.append(key)
                        data_lists[key] = getattr(self, key)
            return data_lists
        
        path = Path("data", "created_datasets", self.dataset_name)
        # Get data from json file
        if path.joinpath(self.dataset_name+'_dataset.json').exists():
            path = path.joinpath(self.dataset_name+'_dataset.json')
            self.data_lists = fetch_from_json()
        # Get data from tensorflow datasets
        elif path.exists():
            self.prebuild_train, self.prebuild_test = tfdsload(self.dataset_name, split=['train', 'test'], as_supervised=True, data_dir=path)
            if sub_sample is not None:
                self.prebuild_train = self.prebuild_train.take(sub_sample)
                self.prebuild_test = self.prebuild_test.take(sub_sample)
        else:
            print("Dataset "+self.dataset_name+" does not exist...")
            exit()
        
    
    def get_values(self, index):
        if hasattr(self, 'data_lists'):
            return [[key, dlist[index]] for key, dlist in self.data_lists.items()]

    def get_data_keys(self):
        if hasattr(self, 'data_key_list'):
            return [self.data_key_list.keys()]

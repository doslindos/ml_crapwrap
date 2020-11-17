from .. import Dataset, split_dataset, nparray, tfdata, duplicate_along, tfpy_func, tfuint8, tfint64, sk_label_encoding, sk_one_hot
from . import normalize

class Preprocess:
    # Every preprocess function should return a tuple with dicts where first instance is training data and second instance is test data
    # Training data and test data dicts should have keys x for data and y for labels

    def __init__(self):
        self.dataset_name = ''
        self.original_data = None
        self.preprocessed_dataset = None
    
    def spotify(
            self, 
            dataset_name, 
            sub_sample=None, 
            data_key='features', 
            label_key='popularity', 
            train_size=0.8, 
            norm_function='normalize_spotify'
            ):
        
        #Fetch dataset and remove duplicates
        dataset = Dataset(dataset_name, sub_sample)
        #dataset.merge_duplicates(data_key)

        #Take lists from dataset with keys
        data = getattr(dataset, data_key)
        
        if hasattr(dataset, 'labels'):
            labels = getattr(dataset, 'labels')
            label_key = 'labels'
        else:
            labels = getattr(dataset, label_key)

        # Store original data
        self.original_data = tuple(zip(data, labels))
        
        data = nparray(data)

        #Normalize selected dimensions
        #data = getattr(normalize, norm_function)(data)
        
        #Change popularity values from scale 0-100 to 0-9
        if label_key == 'popularity':
            label_lines = [*range(10, 100, 10)]
            for i, label in enumerate(labels):
                
                #If label is merged and has a list of different values, use max
                if isinstance(label, list):
                    label = max(label)

                if label >= 90:
                    final_group = 9
                else:
                    for group, line in enumerate(label_lines):
                    
                        if label < line:
                            final_group = group
                            break
                
                labels[i] = final_group
        else:
            label_encoder = sk_label_encoding(labels)
            labels = label_encoder.transform(labels)
            self.label_encoder = label_encoder
        
        x_train, x_test, y_train, y_test = split_dataset(data, labels, train_size)
        
        self.preprocessed_dataset = ({'x':x_train, 'y':y_train}, {'x':x_test, 'y':y_test})

    def mnist(
            self, 
            dataset_name, 
            sub_sample=None,
            norm_function='normalize_image'
            ):
    
        #Fetch dataset and remove duplicates
        self.dataset_name = dataset_name
        dataset = Dataset(dataset_name, sub_sample)
            
        train = dataset.prebuild_train
        test = dataset.prebuild_test
        self.original_data = (train, test)
        
        train = train.map(getattr(normalize, norm_function), num_parallel_calls=tfdata.experimental.AUTOTUNE)
        test = test.map(getattr(normalize, norm_function), num_parallel_calls=tfdata.experimental.AUTOTUNE)
        self.preprocessed_dataset = (train, test)
    
    def mnist_color(
            self, 
            dataset_name, 
            sub_sample=None,
            norm_function='normalize_image'
            ):
    
        #Fetch dataset and remove duplicates
        self.dataset_name = dataset_name
        dataset = Dataset(dataset_name, sub_sample)
            
        train = dataset.prebuild_train
        test = dataset.prebuild_test
        self.original_data = (train, test)
        
        train = train.map(
                    lambda i, l: tfpy_func(
                                    func=duplicate_along,
                                    inp=[i, len(i.shape)-1, l],
                                    Tout=[tfuint8, tfint64]
                                    ), 
                    num_parallel_calls=tfdata.experimental.AUTOTUNE
                    )
        test = test.map(
                    lambda i, l: tfpy_func(
                                    func=duplicate_along,
                                    inp=[i, len(i.shape)-1, l],
                                    Tout=[tfuint8, tfint64]
                                    ), 
                    num_parallel_calls=tfdata.experimental.AUTOTUNE
                    )
        
        train = train.map(getattr(normalize, norm_function), num_parallel_calls=tfdata.experimental.AUTOTUNE)
        test = test.map(getattr(normalize, norm_function), num_parallel_calls=tfdata.experimental.AUTOTUNE)
        self.preprocessed_dataset = (train, test)

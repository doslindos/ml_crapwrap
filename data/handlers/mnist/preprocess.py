from .. import tfdata, normalize_image

class DataPreprocessor:

    def preprocess_set(self, the_set):
        # Preprocesses every instance in the set with normalize image function
        the_set = the_set.map(normalize_image, num_parallel_calls=tfdata.experimental.AUTOTUNE)
        return the_set

    def preprocess(self, dataset):
        if isinstance(dataset, list):
            train, validation, test = dataset
        
            # Preprocess the datasets
            train = self.preprocess_set(train)
            validation = self.preprocess_set(validation)
            test = self.preprocess_set(test)

            return (train, validation, test)
        else:
            return self.preprocess_set(dataset)

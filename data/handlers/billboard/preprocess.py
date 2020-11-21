from .. import tfdata, normalize_spotify_features

class DataPreprocessor:

    def preprocess_set(self, the_set):
        return the_set.map(normalize_spotify_features, num_parallel_calls=tfdata.experimental.AUTOTUNE)

    def preprocess(self, dataset):
        
        if isinstance(dataset, tuple):
            train, validation, test = dataset
            # Preprocess the datasets
            train = self.preprocess_set(train)
            validation = self.preprocess_set(validation)
            test = self.preprocess_set(test)

            return (train, validation, test)
        else:
            return self.preprocess_set(dataset)

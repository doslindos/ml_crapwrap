from .. import tfdata, split_dataset, preprocess_spotify_features
from sklearn.preprocessing import MinMaxScaler
from third_party.scipy.util import print_description
from numpy import array as nparray, unique as npunique
from collections import Counter

class DataPreprocessor:

    def check_unique(self, features):
        uniques, indexes, counts = npunique(features, return_index=True, return_counts=True, axis=0)
        # NOTE try out filter function
        duplicate_indexes = [i for i, dupli in enumerate(features) if i not in indexes]
        print(len(duplicate_indexes))
        print(uniques.shape, indexes)
        print(features.shape)
        return (uniques, duplicate_indexes, indexes)

    def preprocess_features(self, features):
        # Duration to scale 0 to 1
        if not hasattr(self, 'feature_scaler'):
            self.feature_scaler = MinMaxScaler()
            self.feature_scaler.fit(features)
        
        return self.feature_scaler.transform(features)


    def preprocess_labels(self, sample):
        for i, s in enumerate(sample):
            if s < 10:
                sample[i] = 0
            elif s < 20:
                sample[i] = 1
            elif s < 30:
                sample[i] = 2
            elif s < 40:
                sample[i] = 3
            elif s < 50:
                sample[i] = 4
            elif s < 60:
                sample[i] = 5
            elif s < 70:
                sample[i] = 6
            elif s < 80:
                sample[i] = 7
            elif s < 90:
                sample[i] = 8
            else:
                sample[i] = 9

        return sample

    def preprocess(self, dataset):
        # Take features and popularities from the sample
        # Also morph popularities into sets of tens
        features = []
        popularities = []
        for d in dataset:
            features.append(d['features'])
            popularity = d['popularity']

            popularities.append(popularity)
        
        # Cast to float32
        features = nparray(features, dtype="float32")
        
        features, duplicate_indexes, selected_indexes = self.check_unique(features)
        labels = popularities
        # NOTE try out filter
        labels = [l for i, l in enumerate(labels) if i in selected_indexes]
        
        duplicate_instances = []
        for i, d in enumerate(dataset):
            if i in duplicate_indexes:
                duplicate_instances.append(d)

        #print([d['name'] for d in duplicate_instances])
        
        # Apply scaling
        features = self.preprocess_features(features)
        labels = self.preprocess_labels(labels)
        

        # Description
        print_description(features)
        print_description(labels)
        
        # Split dataset
        train, validation, test = split_dataset(features, labels, 0.33, True, 0.15)
        
        # Wrap to tf dataset
        train = tfdata.Dataset.from_tensor_slices((train[0], train[1]))
        test = tfdata.Dataset.from_tensor_slices((test[0], test[1]))
        validate = tfdata.Dataset.from_tensor_slices((validation[0], validation[1]))
        
        return (train, validate, test)

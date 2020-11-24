from .. import tfdata, preprocess_spotify_features, split_dataset
from sklearn.preprocessing import MinMaxScaler
from third_party.scipy.util import print_description
from numpy import array as nparray

class DataPreprocessor:

    def preprocess_features(self, features):
        # Duration to scale 0 to 1
        if not hasattr(self, 'feature_scaler'):
            self.feature_scaler = MinMaxScaler()
            self.feature_scaler.fit(features)
        
        return self.feature_scaler.transform(features)
    
    def preprocess_labels(self, sample):
        # Set decade to be the label
        for i, s in enumerate(sample):
            if s < 10:
                sample[i] = 5
            elif s < 20:
                sample[i] = 6
            elif s < 60:
                sample[i] = 0
            elif s < 70:
                sample[i] = 1
            elif s < 80:
                sample[i] = 2
            elif s < 90:
                sample[i] = 3
            elif s >= 90:
                sample[i] = 4

        return sample

    def preprocess(self, dataset):
        
        # Take features and popularities from the sample
        # Also morph popularities into sets of tens
        features = []
        labels = []
        for d in dataset:
            features.append(d['features'])
            label = d['labels']

            labels.append(int(label))
 
        # Apply scaling
        features = self.preprocess_features(features)
        labels = self.preprocess_labels(labels)
        
        # Cast to float32
        features = nparray(features, dtype="float32")

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

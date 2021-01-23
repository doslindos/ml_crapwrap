from .. import tfdata, split_dataset, preprocess_spotify_features
from sklearn.preprocessing import MinMaxScaler
from third_party.scipy.util import print_description
from numpy import array as nparray, unique as npunique
from collections import Counter
from pickle import dump as pkldump, load as pklload

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

    def preprocess(self, dataset, processed_path, scale=True, balance=True, new_split=False):
        
        processed_path = processed_path.joinpath('processed.pkl')
        if not processed_path.exists() or new_split:
            # Take features and popularities from the sample
            # Also morph popularities into sets of tens
            features = []
            popularities = []
            for d in dataset:
                feat = d['features']
                # Take release year from date and add it to the features
                date = d['release_date']
                if '-' in date:
                    split = date.split('-')
                    if len(split) == 3:
                        y, m, da = split
                    elif len(split) == 2:
                        y, m = split
                    else:
                        print("Not possible", date)

                elif len(date) == 4:
                    y = date
                else:
                    print(date)
                    exit()
            
                # Use only the decade
                y = y[2:]
            
                # Add release year to the features
                feat.append(y)

                # Append to the feature list
                features.append(feat)
            
                # Append to popularity list
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
        
            labels = self.preprocess_labels(labels)
        
            # Split dataset
            datasets = split_dataset(features, labels, 0.33, True, 0.15)
        
            # Store split
            pkldump(datasets, processed_path.open('wb'))

        else:

            # Load stored split
            datasets = pklload(processed_path.open('rb'))
        
        if scale:
            # Apply scaling
            for dataset in datasets:
                features = self.preprocess_features(dataset[0])

        # Description
        #print_description(features)
        #print_description(labels)
        
        # Wrap to tf dataset
        train = tfdata.Dataset.from_tensor_slices((datasets[0][0], datasets[0][1]))
        test = tfdata.Dataset.from_tensor_slices((datasets[1][0], datasets[1][1]))
        validate = tfdata.Dataset.from_tensor_slices((datasets[2][0], datasets[2][1]))
        
        return (train, validate, test)

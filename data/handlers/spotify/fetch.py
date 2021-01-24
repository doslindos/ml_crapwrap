from .. import spotify_api_fetch, mysqldb_fetch, jsonload, rndsample
from collections import Counter

class DataFetcher:

    def load_data(self, sample=None):
        if not self.save_path.exists():
            # Fetch data from mysql
            data, labels = mysqldb_fetch(self.sql_path)
            if sample is not None:
                data = rndsample(data, sample)

            spotify_api_fetch(data, self.save_path)
        
        self.dataset = jsonload(self.save_path.open("r", encoding='utf-8'))

    def get_data(self, sample=None):
        # Wrap the dataset into a Tensorflow Dataset object
        if sample is not None:
            return rndsample(self.dataset, sample)
        else:
            return self.dataset
        
        

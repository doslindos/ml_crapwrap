from .. import Path, spotify_api_fetch, mysqldb_fetch, jsonload, rndsample
from collections import Counter

class DataFetcher:
    
    def __init__(self, ds_name):
        self.dataset_name = ds_name
        # Path to the sql file for mysql fetch
        self.sql_path = Path("data", "handlers", "spotify", "resources", "popularities.sql")
        # Dataset saving path
        self.save_path = Path("data", "handlers", "spotify", "dataset", "spotify_dataset.json")

    def load_data(self):
        if not self.save_path.exists():
            # Fetch data from mysql
            data, labels = mysqldb_fetch(self.sql_path)
            spotify_api_fetch(data, self.save_path)
        
        self.dataset = jsonload(self.save_path.open("r", encoding='utf-8'))

    def get_data(self, sample=None):
        # Wrap the dataset into a Tensorflow Dataset object
        if sample is not None:
            return rndsample(self.dataset, sample)
        else:
            return self.dataset
        
        

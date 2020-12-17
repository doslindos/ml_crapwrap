from .. import Path, spotify_api_fetch, mysqldb_fetch, jsonload, rndsample
from collections import Counter

class DataFetcher:
    
    def __init__(self, h_name, ds_name, source="popularities.sql"):
        self.handler_name = h_name
        self.dataset_name = ds_name
        # Path to the sql file for mysql fetch
        self.sql_path = Path("data", "handlers", "spotify", "resources")
        self.sql_path = self.sql_path.joinpath(source)
        
        # Dataset saving folder
        self.save_folder = Path("data", "handlers", "spotify", "datasets", ds_name)
        
        # DS save path
        save_name = ds_name+"_dataset.json"

        self.save_path = self.save_folder.joinpath(save_name)

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
        
        

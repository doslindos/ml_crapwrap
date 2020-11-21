from .. import Path, spotify_api_fetch, mysqldb_fetch, jsonload, rndsample

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
            sample = rndsample(self.dataset, sample)
        else:
            sample = self.dataset
        
        # Split dataset
        train, validation, test = split_dataset(sample['features'], sample['popularity'], 0.33, True, 0.15)
        
        # Wrap to tf dataset
        train = tfdata.from_tensor_slices((train[0], train[1]))
        test = tfdata.from_tensor_slices((test[0], test[1]))
        train = tfdata.from_tensor_slices((validation[0], validation[1]))

        return (train, validation, test)
        

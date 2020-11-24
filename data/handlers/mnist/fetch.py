from .. import Path, load_with_tfds_load

class DataFetcher:
    
    def __init__(self, ds_name):
        self.ds_name = ds_name
        self.save_path = Path("data", "handlers", ds_name, "dataset")

    def load_data(self):
        # Fetch data from online with tfds_load function
        self.dataset = load_with_tfds_load(self.ds_name, self.save_path, split=['train[:85%]', 'train[85%:]', 'test'], as_supervised=True)

    def get_data(self, sample=None):
        if sample is None:
            # Fetch the full dataset
            return self.dataset
        else:
            # Fetch a random sample
            train = self.dataset[0]
            train.shuffle()
            return train.take(sample)

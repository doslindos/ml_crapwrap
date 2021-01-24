from .. import Path, load_with_tfds_load

class DataFetcher:

    def load_data(self, split):
        # Fetch data from online with tfds_load function
        if split is None:
            split = self.split

        self.dataset = load_with_tfds_load(
                self.handler_name, 
                self.save_path, 
                split=split, 
                as_supervised=True
                )

    def get_data(self, sample=None):
        if sample is None:
            # Fetch the full dataset
            return self.dataset
        else:
            # Fetch a random sample
            train = self.dataset[0]
            train.shuffle()
            return train.take(sample)

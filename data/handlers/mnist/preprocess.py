from .. import tfdata, normalize_image, Path
from .fetch import DataFetcher
from ...util.utils import save_encoders

class DataPreprocessor(DataFetcher):

    
    def __init__(self, h_name, ds_name, split=['train[:85%]', 'train[85%:]', 'test']):
        self.handler_name = h_name
        self.ds_name = ds_name
        self.save_path = Path("data", "handlers", h_name, "datasets", ds_name)
        self.split = split
        self.prepro = normalize_image

        super()
    
    def preprocess_set(self, the_set):
        # Preprocesses every instance in the set with normalize image function
        the_set = the_set.map(self.prepro, num_parallel_calls=tfdata.experimental.AUTOTUNE)
        return the_set

    def preprocess(self, dataset, scale=True, balance=True, new_split=False):
        if not self.save_path.joinpath("encoders.pkl").exists():
            save_encoders(self.save_path, {"Encode": self.prepro, "Type": "Image"})
        if isinstance(dataset, list):
            train, validation, test = dataset
        
            # Preprocess the datasets
            train = self.preprocess_set(train)
            validation = self.preprocess_set(validation)
            test = self.preprocess_set(test)

            return (train, validation, test)
        else:
            return self.preprocess_set(dataset)

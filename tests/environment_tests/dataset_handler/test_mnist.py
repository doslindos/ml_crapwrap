# Hack so that tests are importable in different levels
try:
    from . import DatasetHandlerTester
except:
    from util import DatasetHandlerTester


class MnistHandler(DatasetHandlerTester):
    
    @classmethod
    def setUpClass(cls):
        # Make DataHandlerTester class methods available
        super()

        # Create the mnist handlers
        cls.available_handlers(cls, ['mnist'])
        cls.create_handlers(cls)
    
    def test_data_fetching(self):
        # Use only small split of the dataset
        self.load_function(['train[:2%]', 'train[2%:4%]', 'train[4%:6%]'])
        # Try to fetch the data
        self.fetch_function(None)
    
    @classmethod
    def tearDownClass(cls):
        print("Tearing down")
        cls.destroy_test_datasets(cls)

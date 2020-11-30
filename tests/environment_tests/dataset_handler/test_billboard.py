# Hack so that tests are importable in different levels
try:
    from . import DatasetHandlerTester
except:
    from util import DatasetHandlerTester

class BillboardHandler(DatasetHandlerTester):
    
    @classmethod
    def setUpClass(cls):
        # Make DataHandlerTester class methods available
        super()

        # Create the billboard handlers
        cls.available_handlers(cls, ['billboard'])
        cls.create_handlers(cls)
    
    def test_data_fetching(self):
        print("\nFETCH")
        # Random subsample size for testing
        sample = 300
        # Load only sample of the full data
        self.load_function(sample)
        # Try to fetch the dataset
        self.fetch_function(None)
    
    @classmethod
    def tearDownClass(cls):
        print("Tearing down")
        cls.destroy_test_datasets(cls)

# Hack so that tests are importable in different levels
try:
    from . import DatasetHandlerTester
except:
    from util import DatasetHandlerTester

class SpotifyHandler(DatasetHandlerTester):
    
    @classmethod
    def setUpClass(cls):
        # Make DataHandlerTester class methods available
        super()

        # Create the spotify handlers
        cls.available_handlers(cls, ['spotify'])
        cls.create_handlers(cls)
    
    def test_data_fetching(self):
        # Random subsample size for testing
        sample = 300
        # Load only sample of the full data
        self.load_function(300)
        # Try to fetch the data
        self.fetch_function(None)
    
    @classmethod
    def tearDownClass(cls):
        print("Tearing down")
        cls.destroy_test_datasets(cls)

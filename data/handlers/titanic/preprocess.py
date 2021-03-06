from .. import tfdata, rndsample, jsonload, Path
from ...util.fetchers.kaggle.kaggle_fetcher import KaggleCompetitionDataFetcher

class DataPreprocessor(KaggleCompetitionDataFetcher):

    def __init__(self, h_name, ds_name, source=""):
        self.handler_name = h_name
        self.dataset_name = ds_name
        self.kaggle_competition_name = 'titanic'
        
        # Dataset saving folder
        self.save_folder = Path(Path.cwd(), "data", "handlers", "titanic", "datasets", ds_name)
        print(self.save_folder)
        super()

    def get_data(self, sample=None):
        # Load ds
        self.dataset = jsonload(self.save_path.open("r", encoding='utf-8'))
        if sample is not None:
            return rndsample(self.dataset, sample)
        else:
            return self.dataset

    def preprocess(self, dataset, scale=True, balance=True, new_split=False):
        pass
        #return (train, validate, test)

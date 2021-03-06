from ...data_fetching import kaggle_download, kaggle_competition_download

class KaggleDataFetcher:

    def load_data(self, sample=None):
        if not self.save_path.exists():
            # Download ds
            kaggle_download(self.kaggle_dataset_name, self.save_folder)


class KaggleCompetitionDataFetcher:

    def load_data(self, sample=None):
        if not self.save_folder.exists():
            # Download ds
            kaggle_competition_download(self.kaggle_competition_name, self.save_folder)


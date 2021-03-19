from .. import Path, open_dirGUI
from data.dataset_handler import DatasetHandler

def get_dataset():
    # Fetch the dataset path
    dataset_path = Path(open_dirGUI(Path("data", "handlers")))

    # Load the dataset
    dataset = DatasetHandler(dataset_path.parent.parent.name, dataset_path.name, None)
    dataset.load()

    return dataset.fetch_preprocessed_data()

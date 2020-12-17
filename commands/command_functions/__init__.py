from data import DatasetHandler, data_info
from models import ModelHandler, select_weights, read_prediction_file
from tests.model_tests import test_functions
from utils.functions import run_function
from utils.modules import fetch_model, if_callable_class_function
from tensorflow import config
from pathlib import Path
from GUI.GUI_utils import open_dirGUI, open_fileGUI

def load_data(ds_name, source_file, handler, new_split=False):
    # If handler is not given use dataset name
    if handler is None:
        handler = ds_name

    # Initialize Dataset
    dataset = DatasetHandler(handler, ds_name, source_file)
    # Load the data
    dataset.load(new_split)
    return dataset

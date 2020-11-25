from data import DatasetHandler, data_info
from models import ModelHandler, select_weights, read_prediction_file
from tests.model_tests import test_functions
from utils.functions import run_function
from utils.modules import fetch_model, if_callable_class_function
from tensorflow import config
from pathlib import Path

def load_data(name):
    # Initialize Dataset
    dataset = DatasetHandler(name)
    # Load the data
    dataset.load()
    return dataset

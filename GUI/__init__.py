from functools import partial
from pathlib import Path
from tkinter import Tk, ttk, filedialog, StringVar, IntVar, BooleanVar
from random import sample as random_sample
from sys import exit

from tensorflow import data as tfdata, reshape as tfreshape, is_tensor
from numpy import argmax as npargmax, array as nparray, reshape as npreshape, squeeze as npsqueeze
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from utils.main_utils import run_function, get_dataset_info, fetch_model, dataset_generator, fetch_resource, take_image_screen, get_function_attr_values
from .GUI_utils import build_blueprint, open_dirGUI, open_fileGUI, show_data_tk
from data.setup_functions.data_functions import Preprocess
from data.setup_functions import normalize

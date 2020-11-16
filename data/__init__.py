from pathlib import Path
from collections import Counter
from operator import itemgetter
from json import dump as jsondump, load as jsonload
from sys import exit
from csv import reader as csv_reader

from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from numpy import array as nparray, float32 as npfloat32, append as npappend, save as npsave
from tensorflow_datasets import load as tfdsload
from tensorflow import cast as tfcast, float32 as tffloat32, py_function as tfpy_func, uint8 as tfuint8, int64 as tfint64
from tensorflow import data as tfdata

from .dataset import Dataset
from plot_operations.plot_utils import build_histogram
from sklearn_operations.sklearn_functions import split_dataset
from utils.main_utils import duplicate_along


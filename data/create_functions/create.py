from .. import Path
from .Spotify_functions import SpotifyAPI

def spotify(data, ds_name):
    # Creates a dataset with track_features
    # In:
    #   data:                   list, track ids
    #   ds_name:                str, name for the new dataset

    api = SpotifyAPI()
    dataset_path = api.make_feature_dataset(data, ds_name)


from .. import nparray, npfloat32, npappend, npsave, jsondump, exit, Path
from importlib.util import find_spec
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from collections import Counter

class SpotifyAPI:
    # Handles spotify api features

    def __init__(self):
        if find_spec('credentials'):
            from credentials import Spotify_API_credentials

            # Initializes spotipy api object
            # Uses Spotify account keys defined in credential.py
            cc = SpotifyClientCredentials(
                Spotify_API_credentials['client_id'], 
                Spotify_API_credentials['secret_key']
                )
            self.sp = Spotify(client_credentials_manager=cc)
        else:
            print("Credentials file not found... Check guide Spotify credential!")

    def search_with_name(self, name, limit=10):
        # Make a spotify API search
        # In:
        #   query:                  str, query string
        #   limit:                  int, number of results returned, max = 50
        # Out:
        #   query results:          dict

        return self.sp.search(name, limit)

    def fetch_track_features(self, track_id_list, batch_size=50):
        # The actual function to fetch the data from Spotify API
        # In:
        #   track__id_list:                         list, track spotify id list or dict, where key = track_id, value = label value or list of DIFFERENT label values
        #   batch_size:                             int, defines how big is one batch of track data to be fetched in every call to the API, max = 50
        # Out:
        #   dataset:                                list of dicts, element contains the data for a single track

        def batch_data(labels):
            for i in range(0, len(track_id_list), batch_size):
                if labels:
                    yield list(track_id_list.keys())[i:i+batch_size]
                else:
                    yield track_id_list[i:i+batch_size]
        

        #Fetch track information in batches and create dataset
        dataset = []
        labels = True if isinstance(track_id_list,  dict) else False
        for i, batch in enumerate(batch_data(labels)):
            #Get ids of current batch and fetch info from Spotify API
            batch_track_ids = batch

            feature_results = self.sp.audio_features(batch_track_ids)
            track_results = self.sp.tracks(batch_track_ids)
            if i % 100 == 0:
                print("Batch ", i)

            for j, result in enumerate(feature_results):
                #Parse track information
                track_info = track_results['tracks'][j]
                artists = []
                for artist in track_info['artists']:
                    artists.append({'id':artist['id'], 'name':artist['name']})
            
                #Get track features
                features = [
                    result['duration_ms'],
                    result['key'],
                    result['mode'],
                    result['acousticness'], 
                    result['danceability'], 
                    result['energy'], 
                    result['instrumentalness'], 
                    result['liveness'], 
                    result['loudness'], 
                    result['speechiness'], 
                    result['valence'], 
                    result['tempo'] 
                    ]
            
                # This is the data for every instance stored in the dataset
                track_data = {
                    'name':track_info['name'],
                    'id':track_info['id'],
                    'artists':artists,
                    'popularity':track_info['popularity'],
                    'duration_ms':track_info['duration_ms'],
                    'features':features
                        }
                
                # Add labels to the track data if defined
                tracks = []
                if labels:
                    track_labels = track_id_list[track_data['id']]
                    if isinstance(track_labels, str):
                        track_data['labels'] = track_labels
                    # For multiple labels take most common value
                    elif isinstance(track_labels, list):
                        track_data['labels'] = Counter(track_labels).most_common(1)[0][0]
                
                tracks.append(track_data)
                
                # Add track to dataset
                for track in tracks:
                    dataset.append(track)
    

        return dataset
    
    def make_feature_dataset(self, track_id_list, save_path, filename=None):
        # Takes spotify track id list, fetches data into dataset
        # and saves data in a created folder with trackdilespath filename as a name
        # In:
        #   track_list:                         list, track spotify id list or dict, where key = track_id, value = label or labels list
        #   filename:                           str, name for the dataset_file
        #   save_path:                          Path object, path to saved dataset
        
        if filename is None:
            if save_path.suffix != '.json':
                print("You must either give a filename or put it in the path")
                exit()
            else:
                # Create grandparent folder if not exists
                grandparent_folder = save_path.parent.parent
                if not grandparent_folder.exists():
                    grandparent_folder.mkdir()
                
                # Create parent folder
                parent_folder = save_path.parent
                if not save_path.exists():
                    if not parent_folder.exists():
                        parent_folder.mkdir()
                        print("Directory ",parent_folder.name," created in ",parent_folder.parent,"...")
                
                # If parent folder is found = dataset is created so ask user if it will be overwritten
                else:
                    print("Directory exists!")
                    overwrite = input("Do you want to overwrite existing features?(y/n)")
                    if overerite != 'y':
                        exit()
        
        # Fetch track features
        dataset = self.fetch_track_features(track_id_list)
        
        # Save features dataset to a json file
        with save_path.open('w', encoding='utf8') as jf:
            jsondump(dataset, jf, ensure_ascii=False)

        print("Features have been saved in: ", save_path," ...")
        

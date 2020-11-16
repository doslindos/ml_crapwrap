from .. import SpotifyClientCredentials, Spotify, nparray, npfloat32, npappend, npsave, jsondump, exit, Path
import importlib
if importlib.find_loader('credentials'):
    from credentials import Spotify_API_credentials
    creds_found = True
else:
    creds_found = False

class SpotifyAPI:
    # Handles spotify api features

    def __init__(self):
        # Initializes spotipy api object
        # Uses Spotify account keys defined in credential.py
        if creds_found:
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

        def batch_data(list_type):
            for i in range(0, len(track_id_list), batch_size):
                if labels:
                    yield list(track_id_list.keys())[i:i+batch_size]
                elif list_type == dict:
                    yield track_id_list[i:i+batch_size]
        

        #Fetch track information in batches and create dataset
        dataset = []
        labels = True if isinstance(track_id_list,  dict) else list
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
                        tracks.append(track_data)
                    # For multiple labels make double instances
                    elif isinstance(track_labels, list):
                        for track_label in track_labels:
                            track_copy = track_data.copy()
                            track_copy['labels'] = track_label
                            tracks.append(track_copy)
                else:
                    tracks.append(track_data)
                
                # Add track to dataset
                for track in tracks:
                    dataset.append(track)
    

        return dataset
    
    def make_feature_dataset(self, track_id_list, ds_name):
        # Takes spotify track id list, fetches data into dataset
        # and saves data in a created folder with trackdilespath filename as a name
        # In:
        #   track_list:                         list, track spotify id list or dict, where key = track_id, value = label or labels list
        #   ds_name:                            str, name for the dataset
        #   labels:                             
        # Out:
        #   save_path:                          Path object, path to saved dataset



        dataset = self.fetch_track_features(track_id_list)
        
        save_path = Path("data", "created_datasets") 
        if not save_path.exists():
            save_path.mkdir()

        save_path = save_path / Path(ds_name)
        
        if save_path.exists():
            print("Directory exists!")
            override = input("Do you want to override existing features?(y/n)")
            if override == 'n':
                exit()
        else:
            save_path.mkdir()
            print("Directory ",save_path.name," created in ",save_path,"...")
 
        with save_path.joinpath(ds_name+"_dataset.json").open('w', encoding='utf8') as jf:
            jsondump(dataset, jf, ensure_ascii=False)

        print("Features have been saved in: ", save_path," ...")
        
        return save_path

from .. import spotify_api_fetch, jsonload, Path, tfdata, split_dataset, rndsample

class DataFetcher:
    
    def __init__(self, ds_name):
        self.dataset_name = ds_name
        self.save_path = Path("data", "handlers", "billboard", "dataset", "billboard_dataset.json")

    def load_data(self):
        if not self.save_path.exists():
            json_data = jsonload(Path("data", "handlers", "billboard", "resources", "billboard.json").open('r', encoding='utf-8'))
            
            # Create a data dict with key = track id and value = labels
            data = {}
            for key, value in json_data.items():
                if 'Spotify_track_id' in value.keys():
                    track_id = value['Spotify_track_id']
                    week_ids = value['Week_ids']
                    # Take just year information from the week ids to be used as label
                    years = []
                    months = []

                    for week_id in week_ids:
                    
                        month, day, year = week_id.split("/")
                        months.append(month)
                        # Because years are from 1958-2019 take only last two digits
                        years.append(year[2:])
            
                    # Count how many times the song was on top 100 list in each year
                    labels = years
                
                    if track_id not in data.keys():
                        data[track_id] = labels
                    else:
                        for label in labels:
                            data[track_id].append(label)

            spotify_api_fetch(data, self.save_path)
        
        self.dataset = jsonload(self.save_path.open("r", encoding='utf-8'))


    def get_data(self, sample=None):
        # Wrap dataset into tensorflow dataset object
        if sample is not None:
            return rndsample(self.dataset, sample)
        else:
            return self.dataset
        

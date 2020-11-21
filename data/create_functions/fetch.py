from .. import Path, tfdsload, csv_reader, jsonload, Counter
from .MySql import MySQL_Connector

# Command line callable database creation functions

def tfds_fetch(path):
    # Fetches a dataset with tensorflow-datasets using name as the dataset name
    # Availabel dataset list can be found here https://www.tensorflow.org/datasets/catalog/overview
    
    # Create datasets folder if not exists
    parent_path = Path("data", "created_datasets")
    if not parent_path.exists():
        parent_path.mkdir()
    
    # Crate dataset folder if not exists
    ds_path = Path("data", "created_datasets", path)
    if not ds_path.exists():
        ds_path.mkdir()

    # Load the dataset
    ds = tfdsload(path, data_dir=ds_path)
    return (None, None)

def mysqldb_fetch(path):
    # Fetches a data, label combination from MySQL dataset
    # In:
    #   path:                   str, path to the sql file
    # Out:
    #   (data, labels)          tuple of arrays where data i = label i

    connector = MySQL_Connector()
    path = Path(path)
    if not path.is_absolute():
        path = Path('data', 'data_scripts') / path
    #print(path)
    data = []
    labels = []
    
    # Experimental
    if path.name == 'open_db':
        fetched_data = connector.select_data()
        for di in fetched_data:
            #print(di)
            data.append(di['data'])
            labels.append(di['label'])

    #Read data with .sql file
    elif path.suffix == '.sql':
        with path.open('r') as f:
            sql = f.readlines()
        #Prepare the sql command
        sql_command = ''
        for i, command in enumerate(sql):
            sql_command += command
        
        #Prepare cursor and fetch data
        cursor = connector.db.cursor(dictionary=True)
        cursor.execute(sql_command)
        d = None
        l = None
        for instance in cursor:
            if 'data' in instance.keys():
                data.append(instance['data'])
            elif d is None:
                done = False
                while not done:
                    d = input("Use as data "+str(list(instance.keys())+ ": "))
                    
                    if d not in instance.keys():
                        print(l, " not a key "+str(list(instance.keys())))
                        done = False
                    else:
                        done = True
            else:
                data.append(instance[d])
            
            if 'label' in instance.keys():
                labels.append(instance['label'])
            elif l is None:
                done = False
                while not done:
                    l = input("Use as labels (no labels, press enter) "+str(list(instance.keys()))+": ")
                    if l == '':
                        print("No labels used...")
                        done = True
                    elif l not in instance.keys():
                        print("\n"+l, " not a key "+str(list(instance.keys())))
                        done = False
                    else:
                        done = True

            elif l != '':
                labels.append(instance[l])
        cursor.close()
    
    return (data, labels)

def file_fetch(path):
    print(path)
    
    path = Path(path)
    if not path.is_absolute():
        path = Path('data', 'data_files') / path
    #print(path)
    data = []
    labels = []
    unwanted_characters = ['ï»¿', ';', '*']
    if path.suffix == '.csv':
        reader = csv_reader(path.open('r'), delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                for key in row:
                    for uw in unwanted_characters:
                        key = key.replace(uw, '')
                    labels.append(key)
            else:
                # Check characters
                for i, r in enumerate(row):
                    if isinstance(r, str):
                        for uw in unwanted_characters:
                            r = r.replace(uw, '')
                        row[i] = r
                data.append(row)
    
    if path.suffix == '.json':
        json_data = jsonload(path.open('r', encoding='utf-8'))

        # billboard.json custom functionality
        if path.name == 'billboard.json':
            # Create a data dict with key = track id and value = labels
            data = {}
            total_counter = 0
            double_counter = 0
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
                        #years.append(week.rsplit('/', 1)[-1])
                
                    # Count how many times the song was on top 100 list in each year
                    # Then take the year with most list occurances and use it as a label
                    #label = Counter(months).most_common(1)[0][0]
                    label = months
                    
                    if track_id not in data.keys():
                        total_counter += 1
                        data[track_id] = label
                    else:
                        double_counter += 1
                        print("Double instance... ", key, data[track_id], label)
                        for label in labels:
                            data['track_id'].append(label)
                        #if isinstance(data[track_id], str):
                        #    if label == data[track_id]:
                        #        data[track_id] = [data[track_id], label]
                        #elif isinstance(data[track_id], list):
                        #    data[track_id].append(label)

            print(total_counter, double_counter)
    
    return (data, labels)


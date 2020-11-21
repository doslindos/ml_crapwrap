from .. import tfds_load, Path, input_check, path_check
from .MySql import MySQL_Connector
from .Spotify_functions import SpotifyAPI

def spotify_api_fetch(data, save_path, filename=None):
    # Creates a dataset with track_features
    # In:
    #   data:                   list, track ids
    #   save_path:              Path object, path to save location
    #   filename:               str, name for the save file if None it must be in path
    
    input_check(data, [list, dict], "data in spotify_api_fetch")
    if filename is not None:
        input_check(filename, [str, None], "filename in spotify_api_fetch")
    input_check(save_path, [Path], "save_path in spotify_api_fetch")

    api = SpotifyAPI()
    api.make_feature_dataset(data, filename, save_path)

def load_with_tfds_load(
        dataset_name, 
        save_path, 
        split=['train', 'test'], 
        as_supervised=True
        ):
    # Takes the dataset name, fetches the data and saves it in save_path
    # In:
    #   dataset_name:                   str, name of tensorflow-dataset
    #   save_path:                      Path object, defines saving location
    #   split:                          tfds_load param
    #   as_supervised:                  tfds_load param
    
    input_check(dataset_name, [str], "dataset_name in load_with_tfds_load")
    input_check(save_path, [Path], "save_path in load_with_tfds_load")
    input_check(split, [list], "split in load_with_tfds_load")
    input_check(as_supervised, [bool], "as_supervised in load_with_tfds_load")

    # Create a sub folder "dataset" to store the fetched ds
    if not save_path.parent.exists():
        save_path.parent.mkdir()
    
    # Crate dataset folder if not exists
    if not save_path.exists():
        save_path.mkdir()

    # Load the dataset
    return tfds_load(
            dataset_name, 
            split=split, 
            as_supervised=as_supervised, 
            data_dir=save_path
            )

def mysqldb_fetch(path):
    # Fetches a data, label combination from MySQL dataset
    # In:
    #   path:                   Path object, path to the sql file
    # Out:
    #   (data, labels)          tuple of arrays where data i = label i
    
    def cli_ui_asker(input_msg):
        while not False:
            inp = input(input_msg)
            if inp == '':
                print("No key given...")
                return inp
            elif l not in instance.keys():
                print("\n"+l, " not a key "+str(list(instance.keys())))
            else:
                return inp

    input_check(path, [Path], "path in mysqldb_fetch")
    
    connector = MySQL_Connector()
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
        # Read sql file
        with path.open('r') as f:
            sql = f.readlines()
        #Prepare the sql command
        sql_command = ''
        for i, command in enumerate(sql):
            sql_command += command
        
        #Prepare cursor and fetch data
        cursor = connector.db.cursor(dictionary=True)
        cursor.execute(sql_command)
        
        data_key = 'data'
        label_key = 'label'
        for instance in cursor:
            # Data is found from results
            if data_key in instance.keys():
                data.append(instance[data_key])
            else:
                # Ask the user to define which is used as data
                data_key = cli_ui_asker("Use as data "+str(list(instance.keys())+ ": "))
                if data_key == '':
                    print("No data key give... exiting...")
                    exit()
                else:
                    data.append(instance[d])
            
            if label_key in instance.keys():
                # Label are defined
                labels.append(instance[label_key])
            else:
                # Ask teh user to define what are used as labels

                label_key = cli_ui_asker("Use as labels (no labels, press enter) "+str(list(instance.keys()))+": ")
                if label_key != '':
                    labels.append(instance[l])
        
        cursor.close()
    
    return (data, labels)

def file_fetch(path):
    # This function is not currently in use but it's for reading csv files
    # TODO Write comment

    input_check(path, [Path], "path from file_fetch")
    path_check(path)
    
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
    
    return (data, labels)

def get_jsoned_dataset(path):
    # Fetches the json dataset from path given
    # In:
    #   path:                           Path object, path to dataset
    
    input_check(path, [Path], "path from get_jsoned_dataset")

    if path.exists():
        return jsonload(path.open('r', encoding='utf-8'))
    else:
        print("Couldn't find dataset from ", path)
        exit()

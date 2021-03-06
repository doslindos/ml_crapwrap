from sys import argv, exit
from pathlib import Path

if __name__ == '__main__':
    command = argv[1]
    if command != 'config':
        name = argv[2]

    if command == 'config':
        conffile = Path('configurationstest.ini')
        if not conffile.exists():
            # Create config template
            with conffile.open("w") as f:
                f.write("[Spotify_API_credentials]\nclient_id = \nclient_secret \n\n[MySQL_connector_params]\nhost = \nuser = \n; useing root without password leave this empty.\n; '' or "" doesn't work\npassword = \ndatabase = ")
        else:
            print("Configurations exists!")
            exit()

    if command == 'handler':
        path_to_handler = Path("data", "handlers", name)
        if not path_to_handler.exists():
            # Create a folder for handler
            path_to_handler.mkdir()
            # Create a resource folder
            path_to_handler.joinpath("resources").mkdir()
            # Create fetch and preprocess files
            with path_to_handler.joinpath("fetch.py").open("w") as f:
                f.write("\n\nclass DataFetcher:\n\n\tdef __init__(self, ds_name):\n\t\tself.dataset_name = ds_name\n\n\tdef load_data(self):\n\t\t# Put data loading functions here\n\t\tpass\n\tdef get_data(self):\n\t\t# Put data fetching functions here\n\t\tpass")
            
            with path_to_handler.joinpath("preprocess.py").open("w") as f:
                f.write("from .. import tfdata\n\nclass DataPreprocessor:\n\n\tdef preprocess(self):\n\t\t# Put data preprocessing functions here\n\t\t# Function should return train, validation and test split of data in a Tensorflow Dataset form\n\t\treturn (train, validate, test)")
        
            print("Data handler "+name+" created in ", path_to_handler.parent)
        else:
            print("Handler named ", name, " exists!")
            exit()

    if command == 'model':
        path_to_model = Path("models", name)
        if not path_to_model.exists():
            # Create a folder for handler
            path_to_model.mkdir()
            # Create a resource folder
            path_to_model.joinpath("configurations").mkdir()
            # Create fetch and preprocess files
            with path_to_model.joinpath("model.py").open("w") as f:
                f.write("\n\nclass Model:\n\n\tdef __init__(self, ds_name):\n\t\tpass\n\n\tdef save(self):\n\t\t# Put model saving functions here\n\t\tpass\n\tdef load(self):\n\t\t# Put model loading functions here\n\t\tpass\n\tdef train(self):\n\t\t# Put model training functions here\n\t\tpass\n\tdef run(self, x):\n\t\t# Put model running functions here\n\t\tpass")
            
            with path_to_model.joinpath("README.md").open("w") as f:
                f.write("# Model "+name.upper()+"\n\n## Description")

            print("Model "+name+" created in ", path_to_model.parent)
        else:
            print("Model named ", name, " exists!")
            exit()


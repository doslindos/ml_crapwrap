from sys import argv, exit
from pathlib import Path

if __name__ == '__main__':
    command = argv[1]
    name = argv[2]

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
            
            with path_to_handler.joinpath("process.py").open("w") as f:
                f.write("from .. import tfdata\n\nclass DataPreprocessor:\n\n\tdef preprocess(self):\n\t\t# Put data preprocessing functions here\n\t\t# Function should return train, validation and test split of data in a Tensorflow Dataset form\n\t\treturn (train, validate, test)")
        else:
            print("Handler named ", name, " exists!")
            exit()


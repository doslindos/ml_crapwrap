from .. import tfdata, rndsample, jsonload, jsondump, Path
from ...util.fetchers.kaggle.kaggle_fetcher import KaggleCompetitionDataFetcher

from zipfile import ZipFile
from csv import reader as csv_reader

def resolve_type(column):
    # Check if column has any digit strings
    if any(i.isdigit() for i in column):
        # Add zeros to empty values
        column = ['0' if i == '' else i for i in column]
        # Check if all values are ints
        if all(i.isdigit() for i in column):
            return (column, int)
        else:
            # Check if there are decimal points and that there are no empty space values
            if any('.' in i for i in column) and not any(' ' in i for i in column):
                # Change empty values to 0.0
                column = ['0.0' if i == '' else i for i in column]
                return (column, float)
            else:
                print("Somethings wrong! ", column)
    # No numeric values
    return (column, str)

def change_types(data):
    # Used to change datatypes in titanic data
    # The dataset contains numeric values in strings
    # Change numeric types from strings to numeric
    
    # Seperate by columns
    columns = zip(*data)
    for i, column in enumerate(columns):
        column, i_type = resolve_type(column)
        # If values were numeric change datatypes of the columns values
        if i_type != str:
            columns[i] = list(map(i_type, column))

    # Pack columns back together and return the data
    return list(zip(*columns))

class DataPreprocessor(KaggleCompetitionDataFetcher):

    def __init__(self, h_name, ds_name, source=""):
        self.handler_name = h_name
        self.dataset_name = ds_name
        self.kaggle_competition_name = 'titanic'
        
        # Dataset saving folder
        self.save_folder = Path(Path.cwd(), "data", "handlers", "titanic", "datasets", ds_name)
        print(self.save_folder)
        super()

    def import_data(self):
        titanic_data = ZipFile(Path("data", "titanic.zip"))
        
        # Read train data csv
        with titanic_data.open("train.csv", 'r') as f:
            lines = (line.decode("utf-8") for line in f)
            train = [line for line in csv_reader(lines)]
            # Assign columns and data
            train_columns = train[0]
            train_data = train[1:]
        
        # Read test data csv
        with titanic_data.open("test.csv", 'r') as f:
            lines = (line.decode("utf-8") for line in f)
            test = [line for line in csv_reader(lines)]
                
            test_columns = test[0]
            test_data = test[1:]

        # Read gender data csv
        with titanic_data.open("gender_submission.csv", 'r') as f:
            lines = (line.decode("utf-8") for line in f)
            genders = [line for line in csv_reader(lines)]
            
            gender_columns = genders[0]
            gender_data = genders[1:]
        
        data = {
            'train': (train_columns, train_data), 
            'test': (test_columns, test_data),
            'gender': (gender_columns, gender_data)
            }
            
        # Save features dataset to a json file
        with self.save_folder.joinpath('titanic.json').open('w', encoding='utf8') as f:
            jsondump(data, f, ensure_ascii=False)
                        

    def get_data(self, sample=None):
        # Path to data
        if not self.save_folder.joinpath("titanic.json").exists():
            # Read files from .zip
            self.import_data()
            
        # Load ds
        self.dataset = jsonload(self.save_path.open("r", encoding='utf-8'))
        if sample is not None:
            return rndsample(self.dataset, sample)
        else:
            return self.dataset

    def preprocess(self, dataset, scale=True, balance=True, new_split=False):
        pass
        #return (train, validate, test)

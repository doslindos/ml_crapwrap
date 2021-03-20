from .. import tfdata, rndsample, jsonload, jsondump, Path, make_tfrecords, read_tfrecords, save_encoders
from ...util.fetchers.kaggle.kaggle_fetcher import KaggleCompetitionDataFetcher

from zipfile import ZipFile
from csv import reader as csv_reader

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from numpy import mean as npmean, array as nparray, float32 as npfloat32, append as npappend
from scipy import stats

def resolve_type(column):
    # Check if column has any digit strings
    if any(i.isdigit() for i in column):
        # Add zeros to empty values
        column = ['0' if i == '' else i for i in column]
        # Check if all values are ints
        if all(i.isdigit() for i in column):
            return (column, int)
        
        # Check if there are decimal points and that there are no empty space values
        if any('.' in i for i in column) and not any(' ' in i for i in column):
            # Change empty values to 0.0
            column = ['0.0' if i == '' else i for i in column]
            return (column, float)

    # No numeric values
    return (column, str)

def change_types(data):
    # Used to change datatypes in titanic data
    # The dataset contains numeric values in strings
    # Change numeric types from strings to numeric
    
    # Seperate by columns
    columns = zip(*data)
    processed = []
    for i, column in enumerate(columns):
        column, i_type = resolve_type(column)
        # If values were numeric change datatypes of the columns values
        if i_type != str:
            processed.append(list(map(i_type, column)))
        else:
            processed.append(column)

    # Pack columns back together and return the data
    return list(zip(*processed))
    
def list_to_2d_array(lst):
    lst_2d = []
    for i in range(len(lst)):
        lst_2d.append(lst[i:i+1])
    return lst_2d

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
        titanic_data = ZipFile(self.save_folder.joinpath("titanic.zip"))
        
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
            'train': (train_columns, change_types(train_data)), 
            'test': (test_columns, change_types(test_data)),
            'gender': (gender_columns, change_types(gender_data))
            }
            
        # Save features dataset to a json file
        with self.save_folder.joinpath('titanic.json').open('w', encoding='utf8') as f:
            jsondump(data, f, ensure_ascii=False)
                        

    def get_data(self, sample=None):
        # Path to data
        datapath = self.save_folder.joinpath("titanic.json")
        if not datapath.exists():
            # Read files from .zip
            self.import_data()
            
        # Load ds
        self.unprocessed_dataset = jsonload(datapath.open("r", encoding='utf-8'))
        if sample is not None:
            return rndsample(self.unprocessed_dataset, sample)
        else:
            return self.unprocessed_dataset

    def preprocess(self, dataset, scale=True, balance=True, new_split=False):
        
        save_path = self.save_folder.joinpath("processed")
        print(save_path)
        if not save_path.exists():
            # Preprocess original data
            ds = {}
            for key, data in dataset.items():
                tags = data[0]
                columns = list(zip(*data[1]))
                ds[key] =  {tag: columns[i] for i, tag in enumerate(tags) }
            train = ds['train']
            test = ds['test']

            final_order = ['Pclass', 'Sex', 'Age', 'Alone', 'Fare', 'Cabin', 'Embarked']
            
            encoders = {'Order': final_order, 'Encoders':[]}
            encoders['Encoders'] = (
                    OneHotEncoder(),
                    (OrdinalEncoder(), OneHotEncoder()),
                    MinMaxScaler(),
                    None,
                    MinMaxScaler(),
                    (OrdinalEncoder(), MinMaxScaler()),
                    (OrdinalEncoder(), MinMaxScaler())
                    )
            encoders['Type'] = "Column"

            enc = encoders['Encoders']
            
            # Transform Pclass
            enc[0].fit(list_to_2d_array(train['Pclass']))
            train['Pclass'] = enc[0].transform(list_to_2d_array(train['Pclass'])).toarray()
            test['Pclass'] = enc[0].transform(list_to_2d_array(test['Pclass'])).toarray()
            
            # Transpose pclass and sex
            train['Pclass'] = list(map(list, zip(*train['Pclass'])))
            test['Pclass'] = list(map(list, zip(*test['Pclass'])))
            
            # Transform Sex
            # Ordinal encoding
            enc[1][0].fit(list_to_2d_array(train['Sex'] + test['Sex']))
            
            train['Sex'] = enc[1][0].transform(list_to_2d_array(train['Sex']))
            test['Sex'] = enc[1][0].transform(list_to_2d_array(test['Sex']))
            
            # One hot encoding
            enc[1][1].fit(train['Sex'])
            train['Sex'] = enc[1][1].transform(train['Sex']).toarray()
            test['Sex'] = enc[1][1].transform(test['Sex']).toarray()
            
            train['Sex'] = list(map(list, zip(*train['Sex'])))
            test['Sex'] = list(map(list, zip(*test['Sex'])))
            
            # Transform Age
            # Fill missing ages with mean
            mean = npmean(nparray([i for i in train['Age'] if i != 0.0]))
            train['Age'] = [i if i != 0.0 else mean for i in train['Age']]
            
            # Fill missing ages with mean
            mean = npmean(nparray([i for i in test['Age'] if i != 0.0]))
            test['Age'] = [i if i != 0.0 else mean for i in test['Age']]
            
            enc[2].fit(list_to_2d_array(train['Age'] + test['Age']))
            train['Age'] = enc[2].transform(list_to_2d_array(train['Age']))
            test['Age'] = enc[2].transform(list_to_2d_array(test['Age']))

            # Transform Alone
            # Feature engineer is travelling alone
            train['Alone'] = [0 if sum(with_num) < 1 else 1 for with_num in list(zip(train['SibSp'], train['Parch']))]
            test['Alone'] = [0 if sum(with_num) < 1 else 1 for with_num in list(zip(test['SibSp'], test['Parch']))]
            
            # If under 15 and alone add 1 because higly unlikely
            train['Alone'] = [alone if alone == 0 and train['Age'][i] > 15 else 1 for i, alone in enumerate(train['Alone'])]
            
            # If under 15 and alone add 1 because higly unlikely
            test['Alone'] = [alone if alone == 0 and test['Age'][i] > 15 else 1 for i, alone in enumerate(test['Alone'])]
            
            # Transform Fare
     
            enc[4].fit(list_to_2d_array(train['Fare'] + test['Fare']))
            train['Fare'] = enc[4].transform(list_to_2d_array(train['Fare']))
            test['Fare'] = enc[4].transform(list_to_2d_array(test['Fare']))
            
            # Transform Cabin
            enc[5][0].fit(list_to_2d_array(train['Cabin'] + test['Cabin']))
            train['Cabin'] = enc[5][0].transform(list_to_2d_array(train['Cabin']))
            
            test['Cabin'] = enc[5][0].transform(list_to_2d_array(test['Cabin']))
            enc[5][1].fit(npappend(train['Cabin'],test['Cabin'], 0))
            train['Cabin'] = enc[5][1].transform(train['Cabin'])
            test['Cabin'] = enc[5][1].transform(test['Cabin'])
            
            # Transform Embarked
            enc[6][0].fit(list_to_2d_array(train['Embarked'] + test['Embarked']))
            train['Embarked'] = enc[6][0].transform(list_to_2d_array(train['Embarked']))
            test['Embarked'] = enc[6][0].transform(list_to_2d_array(test['Embarked']))
            
            enc[6][1].fit(npappend(train['Embarked'], test['Embarked'], 0))
            train['Embarked'] = enc[6][1].transform(train['Embarked'])
            test['Embarked'] = enc[6][1].transform(test['Embarked'])

            # zip back together
            survived = train['Survived']
            train_id_name = list(zip(train['PassengerId'], train['Name']))
            train_data = nparray(list(zip(*train['Pclass'], *train['Sex'], train['Age'], train['Alone'], train['Fare'], train['Cabin'], train['Embarked'])), dtype=npfloat32)
            
            desc = stats.describe(train_data)
            print("Min: ", desc.minmax[0])
            print("Max: ", desc.minmax[1])
            print("Mean: ", desc.mean)
            print("Variance: ", desc.variance)
            print("Kurtosis: ", desc.kurtosis) 
            #exit()
            
            # Split the training set to train and validation set
            train_data, validation_data, survived, validation_survived = train_test_split(train_data, survived, test_size=0.33)
            
            test_id_name = list(zip(test['PassengerId'], test['Name']))
            test_data = nparray(list(zip(*test['Pclass'], *test['Sex'], test['Age'], test['Alone'], test['Fare'], test['Cabin'], test['Embarked'])), dtype=npfloat32)


            desc = stats.describe(test_data)
            print("Min: ", desc.minmax[0])
            print("Max: ", desc.minmax[1])
            print("Mean: ", desc.mean)
            print("Variance: ", desc.variance)
            print("Kurtosis: ", desc.kurtosis) 
                
            print(test_data.shape)
            print(train_data.shape)
            # Wrap to tf dataset
            train = tfdata.Dataset.from_tensor_slices((train_data, survived))
            test = tfdata.Dataset.from_tensor_slices((test_data, None))
            validate = tfdata.Dataset.from_tensor_slices((validation_data, validation_survived))
            
            dataset = {
                    "train": train,
                    "validate": validate,
                    "test": test
                    }
            save_path.mkdir()
            make_tfrecords(save_path, dataset)
            save_encoders(save_path, encoders)

        else:
            datasets = read_tfrecords(save_path)
            train = datasets['train']
            validate = datasets['validate']
            test = datasets['test']

        return (train, validate, test)

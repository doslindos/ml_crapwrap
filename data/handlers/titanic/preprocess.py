from .. import tfdata, rndsample, jsonload, jsondump, Path
from ...util.fetchers.kaggle.kaggle_fetcher import KaggleCompetitionDataFetcher

from zipfile import ZipFile
from csv import reader as csv_reader

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, minmax_scale
from sklearn.model_selection import train_test_split
from collections import Counter
from numpy import mean as npmean, array as nparray, float32 as npfloat32
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
        
        # Preprocess original data
        ds = {}
        for key, data in dataset.items():
            tags = data[0]
            columns = list(zip(*data[1]))
            ds[key] =  {tag: columns[i] for i, tag in enumerate(tags) }
            print(ds.keys())
        train = ds['train']
        test = ds['test']
        print(train.keys())
        print(test.keys())

        
        encoders = {}
        encoders['Sex_ordinal'] = OrdinalEncoder().fit(list_to_2d_array(train['Sex'] + test['Sex']))
        # Transform values
        train['Sex'] = encoders['Sex_ordinal'].transform(list_to_2d_array(train['Sex']))
        test['Sex'] = encoders['Sex_ordinal'].transform(list_to_2d_array(test['Sex']))
        
        encoders['Cabin'] = OrdinalEncoder().fit(list_to_2d_array(train['Cabin'] + test['Cabin']))
        # Transform values
        train['Cabin'] = encoders['Cabin'].transform(list_to_2d_array(train['Cabin']))
        test['Cabin'] = encoders['Cabin'].transform(list_to_2d_array(test['Cabin']))
        
        encoders['Embarked'] = OrdinalEncoder().fit(list_to_2d_array(train['Embarked'] + test['Embarked']))
        # Transform values
        train['Embarked'] = encoders['Embarked'].transform(list_to_2d_array(train['Embarked']))
        test['Embarked'] = encoders['Embarked'].transform(list_to_2d_array(test['Embarked']))
        
        # Take a look of ticket values
        tickets = []
        letter_tickets = []
        for ticket in train['Ticket']:
            try:
                ticket = int(ticket)
                tickets.append(ticket)
            except:
                letter_tickets.append(ticket)

        for ticket in test['Ticket']:
            try:
                ticket = int(ticket)
                tickets.append(ticket)
            except:
                letter_tickets.append(ticket)
        
        # Scale
        
        # Fill missing ages with mean
        mean = npmean(nparray([i for i in train['Age'] if i != 0.0]))
        train['Age'] = [i if i != 0.0 else mean for i in train['Age']]
        
        # Feature engineer is travelling alone
        train['Alone'] = [0 if sum(with_num) < 1 else 1 for with_num in list(zip(train['SibSp'], train['Parch']))]
        # If under 15 and alone add 1 because higly unlikely
        train['Alone'] = [alone if alone == 0 and train['Age'][i] > 15 else 1 for i, alone in enumerate(train['Alone'])]
        #train['SibSp'] = minmax_scale(train['SibSp'])
        #train['Parch'] = minmax_scale(train['Parch'])
        
        #train['Pclass'] = minmax_scale(train['Pcalss'])
        #train['Sex'] = minmax_scale(train['Sex'])

        # One hot encode PC and sex
        encoders['Sex_onehot'] = OneHotEncoder().fit(train['Sex'])
        train['Sex'] = encoders['Sex_onehot'].transform(train['Sex']).toarray()
        
        encoders['Pclass'] = OneHotEncoder().fit(list_to_2d_array(train['Pclass']))
        train['Pclass'] = encoders['Pclass'].transform(list_to_2d_array(train['Pclass'])).toarray()
        
        # Transpose pclass and sex
        train['Pclass'] = list(map(list, zip(*train['Pclass'])))
        train['Sex'] = list(map(list, zip(*train['Sex'])))
        
        train['Age'] = minmax_scale(train['Age'])
        train['Fare'] = minmax_scale(train['Fare'])
        train['Cabin'] = minmax_scale(train['Cabin'])
        train['Embarked'] = minmax_scale(train['Embarked'])

        #test['Pclass'] = minmax_scale(test['Pclass'])
        #test['Sex'] = minmax_scale(test['Sex'])
        # One hot encode PC and sex
        #encoders['Sex_onehot'].fit(test['Sex'])
        test['Sex'] = encoders['Sex_onehot'].transform(test['Sex']).toarray()
      
        one_hot_pclass = OneHotEncoder()
        one_hot_pclass.fit(list_to_2d_array(test['Pclass']))
        test['Pclass'] = one_hot_pclass.transform(list_to_2d_array(test['Pclass'])).toarray()
        
        # Transpose pclass and sex
        test['Pclass'] = list(map(list, zip(*test['Pclass'])))
        test['Sex'] = list(map(list, zip(*test['Sex'])))

        # Fill missing ages with mean
        mean = npmean(nparray([i for i in test['Age'] if i != 0.0]))
        print(mean)
        test['Age'] = [i if i != 0.0 else mean for i in test['Age']]
        test['Age'] = minmax_scale(test['Age'])
        
        # Feature engineer is travelling alone
        test['Alone'] = [0 if sum(with_num) < 1 else 1 for with_num in list(zip(test['SibSp'], test['Parch']))]
        # If under 15 and alone add 1 because higly unlikely
        test['Parch'] = [alone if alone == 0 and test['Age'][i] > 15 else 1 for i, alone in enumerate(test['Alone'])]
        #test['SibSp'] = minmax_scale(test['SibSp'])
        #test['Parch'] = minmax_scale(test['Parch'])
        test['Fare'] = minmax_scale(test['Fare'])
        test['Cabin'] = minmax_scale(test['Cabin'])
        test['Embarked'] = minmax_scale(test['Embarked'])
       
        # zip back together
        survived = train['Survived']
        train_id_name = list(zip(train['PassengerId'], train['Name']))
        train_data = nparray(list(zip(*train['Pclass'], *train['Sex'], train['Age'], train['Alone'], train['Fare'], train['Cabin'], train['Embarked'])), dtype=npfloat32)
        
        #print(self.processed_train_data.shape)
        #for p in self.processed_train_data:
        #    for i in p:
        #        print(type(i))

        #    break
        desc = stats.describe(train_data)
        print("Min: ", desc.minmax[0])
        print("Max: ", desc.minmax[1])
        print("Mean: ", desc.mean)
        print("Variance: ", desc.variance)
        print("Kurtosis: ", desc.kurtosis) 
        #exit()
        
        print(train_data.shape)
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
        
        return (train, validate, test)

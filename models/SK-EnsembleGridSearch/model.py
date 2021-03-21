from . import configurations

from .. import handle_init, save_configuration, save_sk_model, load_configuration, load_sk_model, Path, nparray, npprod

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
import time

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from importlib import import_module

from copy import deepcopy

def get_class(module, class_name):
    module = import_module(module)
    return getattr(module, class_name)

class Model:

    def __init__(self, conf_name):
        handle_init(self, conf_name, configurations)
        self.set_conf()

    def set_conf(self):
        self.models = self.c['models']
    
    def save(self, best_estimator):
        path = Path('models/SK-EnsembleGridSearch/saved_models/')
        if not path.exists():
            path.mkdir()
        path = save_sk_model(best_estimator, path.joinpath(self.conf_name))
        save_configuration(self.c, self.conf_name, path)
    
    def load(self, path):
        self.model = load_sk_model(path)
    
    def call_model_attributes(self, attribute, inputs):
        return getattr(self.model, attribute)(**inputs)
    
    def build_model(self, model):
        # Get model
        m = get_class(model['module'], model['model'])
        
        print(model)
        # Handle stacked models with estimators
        if 'params' in model.keys():
            params = deepcopy(model['params'])
            for key, values in params.items():
                if 'estimators' == key:
                    for i, estimator in enumerate(values):
                        c = get_class(estimator['module'], estimator['model'])
                        e = c(**estimator['params'])
                        params['estimators'][i] = (estimator['name'], e)
                        print(model['params']['estimators'][i])
                        print(params['estimators'][i])

            # Init model
            m = m(**params)
        else:
            m = m()

                            
        # Handle stacked models with estimators as params
        search_params = deepcopy(model['search_params'])
        for i, params in enumerate(search_params):
            if 'base_estimator' in params.keys():
                for j, estimator in enumerate(params['base_estimator']):
                    search_params[i]['base_estimator'][j] = get_class(estimator['module'], estimator['model'])(**estimator['params'])
                
        return (m, search_params)

    def train(self, datasets):
        
        if isinstance(datasets, tuple):
            train, validate = datasets
        else:
            print("Validation set not given...")
            exit()

        clfs = []
        acc_pred = {}
        scores=['precision', 'recall']
        for train_data in train.batch(train.cardinality()):
            x, y = train_data
            x = x.numpy()
            y = y.numpy()
            for score in scores:
                for model in self.models:
                    m, search_params = self.build_model(model)
                    print(model)
                    begin = time.time()
                    clf = GridSearchCV(m, search_params, scoring='%s_macro' % score)
                    print("Training model: ", model['model'])
                    print("Scoring: ", score)
                    clf.fit(x, y)
                    clfs.append(clf)
                    print("Trained... ",time.time()-begin)
            
                for i, clf in enumerate(clfs):
                    print(clf)
                    for val_data in validate.batch(validate.cardinality()):
                        validate_x, validate_y = val_data
                        report = classification_report(validate_y, clf.predict(validate_x))
                        print(report)
                        accuracy = accuracy_score(validate_y, clf.predict(validate_x))
                        #clf.fit(validate_x.numpy(), validate_y.numpy())
                        
                        #report2 = classification_report(validate_y, clf.predict(validate_x))
                        #print(report2)
                        #accuracy2 = accuracy_score(validate_y, clf.predict(validate_x))
                        if 'accuracy' not in acc_pred.keys():
                            acc_pred['accuracy'] = accuracy
                            acc_pred['estimator'] = clf
                            acc_pred['best_params'] = clf.best_params_
                        elif acc_pred['accuracy'] < accuracy:
                            acc_pred['accuracy'] = accuracy
                            acc_pred['estimator'] = clf
                            acc_pred['best_params'] = clf.best_params_
            
        self.save(acc_pred['estimator'])
        self.model = acc_pred['estimator']
        print("Training finished...")

    def run(self, x, training=False):
        print(self.model.best_estimator_)
        try:
            return self.model.transform(x)
        except AttributeError:
            try:
                return self.model.predict(x)
            except AttributeError:
                return self.model.predict_proba(x)

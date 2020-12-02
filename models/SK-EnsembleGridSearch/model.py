from . import configurations

from .. import handle_init, save_configuration, save_sk_model, load_configuration, load_sk_model, Path, nparray, npprod

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
import time

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

    def train(self, datasets):
        
        if len(datasets) == 2:
            train, test = datasets
        elif len(datasets) == 3:
            train, validate, test = datasets

        clfs = []
        acc_pred = {}
        scores=['precision', 'recall']
        for train_data in train.batch(train.cardinality()):
            x, y = train_data
            x = x.numpy()
            y = y.numpy()
            for score in scores:
                for model in self.models:
                    begin = time.time()
                    clf = GridSearchCV(model['model'], model['params'], scoring='%s_macro' % score)
                    print("Training model: ", model['name'])
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
        return self.model.transform(x)

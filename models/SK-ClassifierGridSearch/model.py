from . import configurations

from .. import handle_init, save_configuration, save_sk_model, load_configuration, load_sk_model, Path, nparray, npprod

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from third_party.sklearn.sklearn_functions import plot_learning_curve

import time

class Model:

    def __init__(self, conf_name):
        handle_init(self, conf_name, configurations)

    def save(self, best_estimator):
        path = Path('models/SK-ClassifierGridSearch/saved_models/')
        if not path.exists():
            path.mkdir()
        path = save_sk_model(best_estimator, path.joinpath(self.conf_name))
        save_configuration(self.c['params'], self.conf_name, path)
    
    def load(self, path):
        self.model = load_sk_model(path)
    
    def call_model_attributes(self, attribute, inputs):
        return getattr(self.model, attribute)(**inputs)

    def train(self, datasets):
        if isinstance(datasets, tuple):
            train, validate = datasets
        else:
            print("Validation set not given...")
            exit()

        acc_pred = {}
        scores=['f1']
        for train_data in train.batch(train.cardinality()):
            x, y = train_data
            x = x.numpy()
            y = y.numpy()
            from collections import Counter
            print(Counter(y).most_common())
            for score in scores:
                try:
                    begin = time.time()
                    clf = GridSearchCV(
                        self.c['model'], 
                        self.c['params'], 
                        scoring='%s_macro' % score,
                        n_jobs=-2
                        )
                    print("Training model: ", self.c['name'])
                    print("Scoring: ", score)
                    clf.fit(x, y)
                    print("Trained... ",time.time()-begin)
            
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
                except KeyboardInterrupt:
                    exit()

        self.save(acc_pred['estimator'])
        self.model = acc_pred['estimator']
        print("Training finished...")

    def run(self, x, training=False):
        return self.model.predict_log_proba(x)

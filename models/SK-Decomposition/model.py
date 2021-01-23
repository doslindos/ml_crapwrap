from . import configurations

from .. import handle_init, save_configuration, save_sk_model, load_configuration, load_sk_model, Path, nparray, npprod
from sklearn import decomposition

class Model:

    def __init__(self, conf_name):
        handle_init(self, conf_name, configurations)
        self.set_conf()
        if not hasattr(self, 'model'):
            self.model = getattr(decomposition, self.function)(**self.params)

    def set_conf(self):
        self.params = {}
        for key, value in self.c.items():
            if key == 'function':
                self.function = value
            else:
                self.params[key] = value
    
    def save(self):
        path = Path('models/SK-Decomposition/saved_models/')
        if not path.exists():
            path.mkdir()
        path = save_sk_model(self.model, path.joinpath(self.conf_name))
        save_configuration(self.c, self.conf_name, path)
    
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

        for train_data in train.batch(train.cardinality()):
            x, y = train_data
            x = x.numpy()
            y = y.numpy()
            print("Fitting...")
            self.model.fit(x)

        self.save()
        print("Training finished...")

    def run(self, x, training=False):
        return self.model.transform(x)

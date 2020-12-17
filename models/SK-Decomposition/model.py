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

    def train(self, dataset):
        
        if isinstance(dataset, dict):
            x = dataset['x']
            y = dataset['y']
        elif hasattr(dataset, 'as_numpy_iterator'):
            dataset = list(dataset.as_numpy_iterator())
            if isinstance(dataset[0], tuple):
                x, y = zip(*dataset)

        x = nparray(x)
        
        if len(x[0].shape) > 1:
            x = x.reshape((x.shape[0], npprod(x.shape[1:])))
        self.model.fit(x)

        self.save()
        print("Training finished...")

    def run(self, x, training=False):
        if hasattr(x, 'numpy'):
            x = x.numpy()

        if len(x.shape) == 4:
            x = x.reshape((x.shape[0], npprod(x.shape[1:])))
        
        return self.model.transform(x)

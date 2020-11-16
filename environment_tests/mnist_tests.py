from . import create_dataset_test, training_test, testing_test, Namespace
from utils.main_utils import list_files_in_folder

# RUN MNIST TESTS
# Attributes
create_attrs = Namespace(d='mnist', ff='tfds_fetch', cf=None, name='mnist')
train_attrs = Namespace(
                    d='mnist',
                    pf='mnist',
                    m='',
                    c='',
                    batch_size=100,
                    epochs=1,
                    learning_rate=0.001,
                    optimization_function=None,
                    loss_function=None,
                    sub_sample=1000
                    )

# Add model name here if you want it to be tested
models = ['NeuralNetworks', 'SK-Decomposition']

def run_mnist(log_dict):   
    if create_dataset_test(create_attrs, log_dict):
        for model in models:
            for configuration in list_files_in_folder('models/'+model+'/configurations', '*.py'):
                if 'mnist' in configuration:
                    train_attrs.m = model
                    train_attrs.c = configuration
                    training_test(train_attrs, log_dict)
                else:
                    pass

    return log_dict

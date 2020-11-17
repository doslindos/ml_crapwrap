from . import create_dataset_test, training_test, testing_test, Namespace

# RUN MNIST TESTS
# Attributes
# ! d='popularities.sql' can be any sql script in data/data_scripts/ !
# ! data/data_scripts/ will not have a popularities.sql named script in it as default so you have to make it !
# README has a reference to this under Guide example title

create_attrs = Namespace(d='popularities.sql', ff='mysqldb_fetch', cf='spotify', name='spotify')
train_attrs = Namespace(
                    d='spotify',
                    pf='spotify',
                    m='',
                    c='',
                    batch_size=100,
                    epochs=1,
                    learning_rate=0.001,
                    sub_sample=None
                    )
# Add model name here if you want it to be tested
models = ['NeuralNetworks', 'SK-Decomposition']

def run_spoti(log_dict):   
    if create_dataset_test(create_attrs, log_dict):
        for model in models:
            for configuration in list_files_in_folder('models/'+model+'/configurations', '*.py'):
                if 'spotify' in configuration:
                    train_attrs.m = model
                    train_attrs.c = configuration
                    training_test(train_attrs, log_dict)
                else:
                    pass

    return log_dict

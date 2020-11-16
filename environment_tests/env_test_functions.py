from . import main, traceback

def log_t(logger, main_key, ds, model, conf, msg):
    # Create a dict for datasests
    if ds not in list(logger[main_key].keys()):
        logger[main_key][ds] = {}
    ds_log = logger[main_key][ds]
    
    # Create a dict for models
    if model not in list(ds_log.keys()):
        ds_log[model] = {}
    model_log = ds_log[model]

    # Add configuration to dict
    model_log[conf] = msg

def create_dataset_test(attrs, log=None):
    # Build dataset MNIST
    #print(attrs)
    try:
        print("Trying to create ",attrs.name," dataset...")
        main.create_dataset(attrs)
        print("MNIST dataset creation successful...")
        if log is not None:
            log['Create datasets'][attrs.name] = 'Success!'
        return True
    
    except:
        print("Failed to create MNIST dataset...")
        traceback.print_exc()
        if log is not None:
            log['Create datasets'][attrs.name] = 'Failed...'
        return False

def training_test(attrs, log=None):
    # Test training functions
    try:
        # Train MNIST NN model
        print("Testing ", attrs.m," model with ",attrs.c," configuration training function on ",attrs.d," dataset...")
        main.train_model(attrs)
        print("Training successful...")
        if log is not None:
            log_t(log, 'Train datasets', attrs.d, attrs.m, attrs.c, 'Success!')
        return True

    except:
        print("Failed to train model...")
        traceback.print_exc()
        if log is not None:
            log_t(log, 'Train datasets', attrs.d, attrs.m, attrs.c, 'Failed...')
        return False

def testing_test(attrs, log=None):
    # Test testing functions on MNIST NN model
    try:
        # Test MNIST NN model classification function
        print("Testing ",attrs.d, attrs.m, attrs.test," test function...")

        print("Test function test successful!")
        if log is not None:
            log_t(log, 'Test datasets', attrs.d, attrs.m, attrs.c, 'Success!')
        return True

    except:
        print("Failed ",attrs.d, attrs.m, attrs.test," function...")
        traceback.print_exc()
        if log is not None:
            log_t(log, 'Test datasets', attrs.d, attrs.m, attrs.c, 'Failed...')
        return False


from . import load_data, run_function, ModelHandler

def train_model(parsed):
    # Creates a model user is defined

    dataset = load_data(parsed.d)

    train, validation, test = dataset.fetch_preprocessed_data(parsed.sub_sample)
    
    # Retrieve configuration from user input arguments
    # If it is not defined configuration with _basic ending is used
    # TODO
    # Rewrite
    if parsed.c is None:
        conf = parsed.d+'_basic'
    else:
        conf = parsed.c
    
    #Initialize Trainer
    model_handler = ModelHandler((train, validation, test), parsed.m, conf)
    #Run training
    model_handler.train(parsed)


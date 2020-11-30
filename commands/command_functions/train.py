from . import load_data, run_function, ModelHandler

def train_model(parsed):
    # Creates a model user is defined

    dataset = load_data(parsed.ds, parsed.s, parsed.dh)

    train, validation, test = dataset.fetch_preprocessed_data(parsed.sub_sample)
    
    #Initialize Trainer
    model_handler = ModelHandler((train, validation, test), parsed.m, conf)
    #Run training
    model_handler.train(parsed)


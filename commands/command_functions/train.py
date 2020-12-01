from . import load_data, run_function, ModelHandler

def train_model(parsed):
    # Creates a model user is defined

    dataset = load_data(parsed.ds, parsed.s, parsed.dh)

    train, validation, test = dataset.fetch_preprocessed_data(parsed.sub_sample)
    
    # Initialize model handler
    model_handler = ModelHandler((train, validation, test), parsed.m, parsed.c)
    # Run training
    model_handler.train(parsed)


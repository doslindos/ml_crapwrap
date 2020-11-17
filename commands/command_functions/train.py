from . import Preprocess, run_function, models

def train_model(parsed):
    # Creates a model user is defined

    # Fetch user defined preprocess function from input params
    if parsed.pf is None:
        preprocess_function = parsed.d
    else:
        preprocess_function = parsed.pf
    
    # Initialize Preprocess class
    prep_pipe = Preprocess()
    # Run the preprocessing function
    run_function(
            prep_pipe, 
            preprocess_function, 
            {'dataset_name':parsed.d, 'sub_sample':parsed.sub_sample}
            )
    
    # Retrieve configuration from user input arguments
    # If it is not defined configuration with _basic ending is used
    # TODO
    # Rewrite
    if parsed.c is None:
        conf = parsed.d+'_basic'
    else:
        conf = parsed.c
    
    #Initialize Trainer
    model_handler = models.ModelHandler(prep_pipe.preprocessed_dataset, parsed.m, conf)
    #Run training
    model_handler.train(parsed)


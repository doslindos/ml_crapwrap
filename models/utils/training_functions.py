from .. import tf

# Training handling function

def parse_sample(batch, onehot=True):
    # Parses tensorflow dataset object sample
    # In:
    #   batch:                      tensorflow Dataset tensor, contains data - label pairs
    #   onehot:                     bool, if true convert labels to one_hot_labels
    # Out:
    #   (x, y)                      tuple, (tensorflow Tensor, tensroflow Tensor)

    if isinstance(batch, dict):
        x = batch['x']
        y = batch['y']
    elif isinstance(batch, tuple):
        x, y = batch
    else:
        print("Batch type not recognized... Check models/utils/util_functions.py parse_sample function")
        exit()

    if onehot:
        y = tf.one_hot(y, 10)

    #Cast to same datatype if not already
    if x.dtype != y.dtype:
        y = tf.cast(y, x.dtype)
                
    #Reshape if in wrong form
    if len(x.shape) == 1:
        x = tf.reshape(x, [-1, x.shape[0]])
    if len(y.shape) == 1:
        y = tf.reshape(y, [y.shape[0], -1])

    return (x, y)

def tf_training_loop(
        dataset, 
        model, 
        loss_function, 
        optimization_function, 
        optimizer, 
        epochs=10, 
        onehot=False,
        autoencoder=False
        ):

    # The training loop
    # In:
    #   dataset:                    Tensorflow dataset object
    #   model:                      Model object
    #   loss_function:              Loss function from train_operations/loss_functions.py
    #   optimization_function:      Optimization function from train_operations/optimization.py
    #   optimizer:                  Tensorflow optimizer object
    #   epoch:                      int, how many times is the model trained with the whole dataset
    print("Training starts...")
    
    for epoch in range(epochs):
        for step, batch_x in enumerate(dataset):
            
            x, y = parse_sample(batch_x, onehot=onehot)
            if autoencoder:
                y = x

            loss = optimization_function(
                    model, 
                    x, 
                    y, 
                    loss_function, 
                    optimizer, 
                    training=True
                    )
        
        print("Epoch ",epoch," loss: ", loss)
    print("Training finished...")

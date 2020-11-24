from .. import tf

# Training handling function

def parse_sample(batch, output_shape, onehot=True):
    # Parses tensorflow dataset object sample
    # In:
    #   batch:                      tensorflow Dataset tensor, contains data - label pairs
    #   onehot:                     bool, if true convert labels to one_hot_labels
    # Out:
    #   (x, y)                      tuple, (tensorflow Tensor, tensroflow Tensor)
    
    # Get x and y from batch
    if isinstance(batch, dict):
        x = batch['x']
        y = batch['y']
    elif isinstance(batch, tuple):
        x, y = batch
    else:
        print("Batch type not recognized... Check models/utils/util_functions.py parse_sample function")
        exit()
    
    # If input is not a tensor
    if not hasattr(x, 'numpy'):
        x = tf.convert_to_tensor(x)

    # One hot encoding
    if onehot:
        if output_shape is not None:
            if len(output_shape) == 1:
                y = tf.one_hot(y, output_shape[0], dtype=x.dtype)
            else:
                print("Your output shape is in ", len(output_shape), " dimensions. It should be in only 1 dim")
                exit()
        else:
            print("You are using one hot encoding without defined 'output_shape' in configuration...")
            exit()
    
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
        train, 
        validation,
        model, 
        loss_function, 
        optimization_function, 
        optimizer, 
        epochs=10, 
        onehot=False,
        autoencoder=False,
        debug=False
        ):

    # The training loop
    # In:
    #   train:                      Tensorflow dataset object
    #   validation:                 Tensorflow dataset object or None
    #   model:                      Model object
    #   loss_function:              Loss function from train_operations/loss_functions.py
    #   optimization_function:      Optimization function from train_operations/optimization.py
    #   optimizer:                  Tensorflow optimizer object
    #   epoch:                      int, how many times is the model trained with the whole dataset
    print("Training starts...")
    
    if 'output_shape' in model.c.keys():
        output_shape = model.c['output_shape']
    else:
        output_shape = None

    for epoch in range(epochs):
        print(train.cardinality())
        for step, batch_x in enumerate(train):
            x, y = parse_sample(batch_x, output_shape, onehot)
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
            
            if validation is not None:
                total_val_loss = 0
                for batch in validation.batch(validation.cardinality().numpy()):
                    x, y = parse_sample(batch, output_shape, onehot)
                    if autoencoder:
                        y = x

                    validation_loss = loss_function(model.run(x, training=False), y)
                    total_val_loss =+ validation_loss.numpy()

            print("Batch ",step," loss: ", total_val_loss)
    print("Training finished...")

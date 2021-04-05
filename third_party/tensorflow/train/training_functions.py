from .. import tf
from numpy import set_printoptions

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
        epochs=1, 
        onehot=False,
        autoencoder=False,
        debug=False,
        validation_batch=None
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
    
    set_printoptions(precision=3)
    
    if 'output_shape' in model.c.keys():
        output_shape = model.c['output_shape']
    else:
        output_shape = None

    # Set onehot encoding to False if autoencoder
    if autoencoder:
        onehot = False
        train_metric = None
        validation_metric = None
    else:
        # Initialize accuracy metrics
        train_metric = tf.keras.metrics.Accuracy()
        validation_metric = tf.keras.metrics.Accuracy()

    for epoch in range(epochs):
        # Reset the metric state
        if train_metric is not None:
            train_metric.reset_states()

        for step, batch_x in enumerate(train):
            print("Batch: ", step)
            x, y = parse_sample(batch_x, output_shape, onehot)
            if autoencoder:
                y = x
        
            output, loss = optimization_function(
                    model, 
                    x, 
                    y, 
                    loss_function, 
                    optimizer, 
                    training=True
                    )
            
            if not autoencoder:
                train_metric.update_state(tf.argmax(y, 1), tf.argmax(output, 1))
                print("Overall training accuracy: ", train_metric.result().numpy(), " loss: ", loss.numpy())
            else:
                print("Training batch loss: ", loss.numpy())

            if validation is not None:
                total_val_loss = 0
                if validation_metric is not None:
                    validation_metric.reset_states()
                if not validation_batch:
                    validation_batch = validation.cardinality().numpy()
                for batch in validation.batch(validation_batch):
                    x, y = parse_sample(batch, output_shape, onehot)
                    if autoencoder:
                        y = x
                    val_out = model.run(x, training=False)
                    validation_loss = loss_function(val_out, y)
                    total_val_loss =+ validation_loss.numpy()
                    if validation_metric is not None:
                        validation_metric.update_state(
                                tf.argmax(y, 1), 
                                tf.argmax(val_out, 1)
                                )

                if not autoencoder:
                    print("Validation accuracy", validation_metric.result().numpy(), " loss: ", total_val_loss)
                else:
                    print("Validation loss: ", total_val_loss)

    print("Training finished...")

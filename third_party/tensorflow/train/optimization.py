from .. import tf, get_weights

def classifier(model_object, x, y, loss_function, optimizer, training=True):

    with tf.GradientTape() as g:
        #Feed input to model
        output = model_object.run(x, training)

        #Calculate loss
        loss = loss_function(output, y)
    
    #Get models trainable variables
    if not hasattr(model_object, 'trainable_vars'):
        # Store trainable variables to the model object during the training session
        model_object.trainable_vars = []
        ws = model_object.weights
        bs = model_object.bias
        for layer, w in ws.items():
            if w[0]:
                model_object.trainable_vars += get_weights(w[1])
                if layer_name in bs.keys() and bs[layer][0]:
                    model_object.trainable_vars += get_weights(bs[layer_name][1])
    
    gradients = g.gradient(loss, model_object.trainable_vars)
    if training:
        optimizer.apply_gradients(zip(gradients, model_object.trainable_vars))

    return (output, loss)


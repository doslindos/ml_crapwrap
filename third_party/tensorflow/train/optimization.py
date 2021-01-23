from .. import tf

def classifier(model_object, x, y, loss_function, optimizer, training=True):

    with tf.GradientTape() as g:
        #Feed input to model
        output = model_object.run(x, training)

        #Calculate loss
        loss = loss_function(output, y)
    
    #Get models trainable variables
    trainable_vars = []
    ws = model_object.weights
    bs = model_object.bias
    for layer_name, w in ws.items():
        if layer_name in bs.keys() and bs[layer_name][0]:
            trainable_vars += w[1] + bs[layer_name][1]
        else:
            if w[0]:
                trainable_vars += w[1]
    
    gradients = g.gradient(loss, trainable_vars)
    if training:
        optimizer.apply_gradients(zip(gradients, trainable_vars))

    return (output, loss)


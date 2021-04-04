from ... import tf
from ..util import check_dtypes_match

# Convolutional

def conv_layer(
        x, 
        weight,
        strides=[1, 2, 2, 1],
        padding='SAME',
        pooling=None,
        bias=None, 
        activation=None,
        batch_norm=None, 
        dropout=None, 
        training=False,
        transpose_shape=()
        ):

    # Convolutional layer
    # In:
    #   x:                          tensorflow Tensor, input data
    #   weight:                     tensorflow Variable, weight
    #   bias:                       tensorflow Variable, bias or None if not used
    #   activation:                 str, name of the class tf.nn function or None if not used
    #   dropout:                    float, value to dropout function or None if not used
    #   training:                   bool
    #   transpose_shape:            tuple, output shape of the transpose layer
    # Out:
    #   return:                     tensorflow Tensor, layer output
    
    # Compare input and weight dtypes
    x = check_dtypes_match(x, weight)
    
    def batch_norm(x):
        #https://stackoverflow.com/questions/46989256/batch-wise-batch-normalization-in-tensorflowrint(pred.shape)
        if len(x.shape) == 2:
            axes = [0]
        elif len(x.shape) == 4:
            axes = [0, 1, 2]
        batch_mean, batch_var = tf.nn.moments(x, axes=axes)
        x = tf.subtract(x, batch_mean)
        x = tf.divide(x, tf.sqrt(batch_var) + 1e-6)
        return x

    if not transpose_shape:
        x = tf.nn.conv2d(x, weight, strides=strides, padding=padding)
        
        if batch_norm is not None and training:
            x = batch_norm(x)

        if bias is not None:
            x = tf.add(x, bias)
    else:
        if bias is not None:
            x = tf.add(x, bias)
        if batch_norm is not None and training:
            x = batch_norm(x)
        
        x = tf.nn.conv2d_transpose(x, weight, transpose_shape, strides=strides, padding=padding)
    
    if pooling is not None:
        ksize = [1, pooling[0], pooling[1], 1]
        x = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

    if activation is not None:
        x = getattr(tf.nn, activation)(x)

    if training and dropout is not None:
        x = getattr(tf.nn, 'dropout')(x, rate=dropout)
    
    return x


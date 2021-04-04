from ... import tf
from ..util import check_dtypes_match

# Dense layer

def dense_layer(x, weight, bias=None, activation=None, dropout=None, training=False, transpose=False):
    # Dense layer
    # In:
    #   x:                          tensorflow Tensor, input data
    #   weight:                     tensorflow Variable, weight
    #   bias:                       tensorflow Variable, bias or None if not used
    #   activation:                 str, name of the class tf.nn function or None if not used
    #   dropout:                    float, value to dropout function or None if not used
    #   training:                   bool
    # Out:
    #   return:                     tensorflow Tensor, layer output
    
    # Compare input and weight dtypes
    x = check_dtypes_match(x, weight)
    
    if not transpose:
        x = tf.matmul(x, weight)
        if bias is not None:
            x = tf.add(x, bias)
    else:
        if bias is not None:
            x = tf.add(x, bias)
        x = tf.matmul(x, weight)
    
    if activation is not None:
        x = getattr(tf.nn, activation)(x)

    if training and dropout is not None:
        x = getattr(tf.nn, 'dropout')(x, rate=dropout)
    
    return x


from .. import tf, npprod

def check_dtypes_match(x, weight):
    
    if x.dtype != weight.dtype:
        x = tf.cast(x, weight.dtype)
    return x

# Weight creation

def create_weights(inp, init_function='RandomNormal', dtype=tf.float32):
    # Create model weight variables
    # In:
    #   inp:                        list, [int, int] or if init = False: numpy array
    #   init_function:              str, function name in tf.initializers class
    #   dtype:                      str, datatype name
    # Out:
    #   return:                     tensorflow Variable
    
    return tf.Variable(getattr(tf.initializers, init_function)()(inp, dtype=dtype))

def create_bias(inp, dtype=tf.float32):
    # Create model bias variables
    # In:
    #   inp:                        list, [int] or if init = False: numpy array
    #   init:                       bool
    #   dtype:                      str, datatype name
    # Out:
    #   return:                     tensorflow Variable
    
    return tf.Variable(tf.zeros(inp, dtype=dtype))

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

# Recurrent cells

def Naive_LSTM_cell(new_input, last_output, weights, biases):
    # Every gate works like this:
    # First add matrix multiplication of input and input weights with
    # matrix multiplication of last output and last output weights
    # Next add bias if used to the output of last step
    # Finally apply activation function

    # In:
    #   new_input:                      Tensor, new input fed to the model
    #   last_output:                    Tensor, final output of the last iteration of the cell
    #   weights:                        dict, containing both weights of all gates
    #   bias:                           dict, containing the biases of all gates
    # Out:

    # Input gate
    x = tf.add(tf.matmul(new_input, weights['input_i']), tf.matmul(last_output, weight['input_o']))
    if bias['input']:
        x = tf.add(x, bias['input'])
    input_gate_output = tf.sigmoid(x)

    # Forget gate
    x = tf.add(tf.matmul(new_input, weight['forget_i']), tf.matmul(last_output, weight['forget_o']))
    if bias['forget']:
        x = tf.add(x, bias['forget'])
    forget_gate_output = tf.sigmoid(x)
    
    # Update gate
    x = tf.add(tf.matmul(new_input, weight['update_i']), tf.matmul(last_output, weight['update_o']))
    if bias['update']:
        x = tf.add(x, bias['update'])
    update_gate_output = tf.tanh(x)
    
    # Output gate
    x = tf.add(tf.matmul(new_input, weights['output_i']), tf.matmul(last_output, weight['output_o']))
    if bias['output']:
        x = tf.add(x, bias['output'])
    output_gate_output = tf.sigmoid(x)

    cell_state = forget_gate_output * last_state + input_gate_output * update_gate_output
    cell_output = output_gate_output * tf.tanh(cell_state)

    return (cell_output, cell_state)

def optimized_LSTM_cell(new_input, output_state, weights, biases, transpose=False):
    
    last_output, last_state = output_state

    if not transpose:
        all_gates = tf.add(
            tf.matmul(new_input, weight['ih']), 
            tf.matmul(last_output, weight['hh'])
            )
    else:
        all_gates = tf.add(
            tf.matmul(new_input, weights['hh']), 
            tf.matmul(last_output, weights['hh'])
            )

    if bias['b']:
        all_gates = tf.add(x, biases['b'])

    input_gate, forget_gate, update_gate, output_gate = tf.split(all_gates, 4, 1)
    
    input_gate_output = tf.sigmoid(input_gate)
    forget_gate_output = tf.sigmoid(forget_gate)
    update_gate_output = tf.tanh(update_gate)
    output_gate_output = tf.sigmoid(output_gate)
    
    cell_state = forget_gate_output * last_state + input_gate_output * update_gate_output
    cell_output = output_gate_output * tf.tanh(cell_state)

    return (cell_output, cell_state)

def symmetric_LSTM_cell(new_input, last_output, weights, biases, transpose=False):

    cell_output, cell_state = optimized_LSTM_cell(new_input, last_output, transpose)

    cell_output = tf.matmul(cell_output, weight['symmetric'])
    if biases['symmetric']:
        cell_output = tf.add(cell_output, biases['symmetric'])

    return (cell_output, cell_state)

# Recurrent

def recurrent_layer(
        x, 
        weight, 
        bias=None, 
        activation=None, 
        dropout=None, 
        training=False, 
        transpose=False
        ):

    # Recurrent layer
    # In:
    #   x:                          tensorflow Tensor, input data
    #   weight:                     tensorflow Variable, weight
    #   bias:                       tensorflow Variable, bias or None if not used
    #   activation:                 str, name of the class tf.nn function or None if not used
    #   dropout:                    float, value to dropout function or None if not used
    #   training:                   bool
    # Out:
    #   return:                     tensorflow Tensor, layer output
    
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

# Recurrent 

from ... import tf
from ..util import check_dtypes_match

# Recurrent cells

def Naive_LSTM_cell(new_input, last_output, weights, biases, transpose=False):
    # Every gate works like this:
    # First add matrix multiplication of input and input weights with
    # matrix multiplication of last output and last output weights
    # Next add bias if used to the output of last step
    # Finally apply activation function

    # Implemented from
    # https://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-in-pytorch-lstms-in-depth-part-1/

    # In:
    #   new_input:                      Tensor, new input fed to the model
    #   last_output:                    Tensor, final output of the last iteration of the cell
    #   weights:                        dict, containing both weights of all gates
    #   bias:                           dict, containing the biases of all gates
    # Out:

    # Input gate
    x = tf.add(tf.matmul(new_input, weights['input']['i']), tf.matmul(last_output, weights['input']['h']))
    if biases is not None:
        x = tf.add(x, biases['input'])
    input_gate_output = tf.sigmoid(x)

    # Forget gate
    x = tf.add(tf.matmul(new_input, weights['forget']['i']), tf.matmul(last_output, weights['forget']['h']))
    if biases is not None:
        x = tf.add(x, biases['forget'])
    forget_gate_output = tf.sigmoid(x)
    
    # Update gate
    x = tf.add(tf.matmul(new_input, weights['update']['i']), tf.matmul(last_output, weights['update']['h']))
    if biases is not None:
        x = tf.add(x, biases['update'])
    update_gate_output = tf.tanh(x)
    
    # Output gate
    x = tf.add(tf.matmul(new_input, weights['output']['i']), tf.matmul(last_output, weights['output']['h']))
    if biases is not None:
        x = tf.add(x, biases['output'])
    output_gate_output = tf.sigmoid(x)

    cell_state = forget_gate_output * last_state + input_gate_output * update_gate_output
    cell_output = output_gate_output * tf.tanh(cell_state)

    return (cell_output, cell_state)

def optimized_LSTM_cell(new_input, output_state, weights, biases, transpose=False):
    
    last_output, last_state = output_state

    if not transpose:
        all_gates = tf.add(
            tf.matmul(new_input, weights['ih']), 
            tf.matmul(last_output, weights['hh'])
            )
    else:
        all_gates = tf.add(
            tf.matmul(new_input, weights['hh']), 
            tf.matmul(last_output, weights['hh'])
            )
    
    if 'b' in biases.keys():
        all_gates = tf.add(all_gates, biases['b'])

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

    cell_output = tf.matmul(cell_output, weights['symmetric'])
    if biases['symmetric']:
        cell_output = tf.add(cell_output, biases['symmetric'])

    return (cell_output, cell_state)


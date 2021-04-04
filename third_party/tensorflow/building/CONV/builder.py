from ..util import create_weights, create_bias

def conv_transpose_weights_creation_loop(weights_list, reversed_bs):
    weights = []
    bias = []
    
    for i, w in enumerate(weights_list):
        w_var = w
        if reversed_bs[i] is not None:
            b_var = reversed_bs[i]
        else:
            b_var = None

        weights.append(w_var)
        bias.append(b_var)

    return (weights, bias)

def conv_weights_creation_loop(kernel_size, filters, use_bias, weight_dtype, transpose):
    # Loop to create convolutional weights
    # In:
    #   filters:                    list, From model configurations.py 'Convo' 'filters'
    #   kernel_size:                list, From model configurations.py 'Convo' 'kernel_size'
    #   use_bias:                   bool, use biases
    #   weight_dtype:               str, datatype used with weights
    #   tranpose:

    # Create "connections" between every layer
    # Configurations 'weights' represents number of units in a hidden layer,
    # therefore the number of connections between layers is 'weights' list lenght - 1
    
    weights = []
    bias = []
    connections = len(filters)-1
    
    for connection in range(connections):
        if connection < connections:
            #print(connection, connections)
            
            # Define conv weight kernel and filter sizes
            k1 = kernel_size[connection][0]
            k2 = kernel_size[connection][1]
            if not transpose:
                f1 = filters[connection]
                f2 = filters[connection+1]
            else:
                f1 = filters[connection+1]
                f2 = filters[connection]
            # Create weights
            w_var = create_weights([k1, k2, f1, f2], dtype=weight_dtype)
            
            if use_bias:
                # Create bias
                b_var = create_bias([f2], dtype=weight_dtype)
            else:
                b_var = None
        
            weights.append(w_var)
            bias.append(b_var)
        else:
            break

    return (weights, bias)
    
def initialize_conv_layer(
        layer_name, 
        input_dtype, 
        conf, 
        weights, 
        bias, 
        transpose
        ):
    # Create weights for the CONV layer if not made
    # In:
    #   conf:                   dict, configuration
    # Out:
    #   (weigths, bias):        (dict, dict) modified weights dicts
    if not layer_name in list(weights.keys()):
        
        # Use reversed weights
        if isinstance(transpose, str) and 'kernel_sizes' not in conf.keys():
            if conf['use_bias']:
                reversed_bs = list(reversed(bias[conf['transpose']][1]))
            else:
                reversed_bs = None
            #print([i.shape for i in weights[conf['transpose']][1]])       
            cws, cbs = conv_transpose_weights_creation_loop(
                list(reversed(weights[conf['transpose']][1])),
                reversed_bs,
                )
            trainable_vars = False
        # Create new weights
        else:
            # Create new weights
            cws, cbs = conv_weights_creation_loop(
                            conf['kernel_sizes'],
                            conf['filters'],
                            conf['use_bias'], 
                            input_dtype,
                            transpose
                            )
            trainable_vars = True

        weights[layer_name] = (trainable_vars, cws)
        bias[layer_name] = (trainable_vars, cbs)

    return (weights, bias)

from ..util import create_weights, create_bias

def LSTM_weights_creation_loop(
        unit_list, 
        use_bias, 
        weight_dtype, 
        cell,
        features,
        transpose=False
        ):
    # Loop to create dense weights
    # In:
    #   weights_list:               list, From model configurations.py 'Dense' 'weights'
    #   use_bias:                   bool, use biases
    #   weight_dtype:               str, datatype used with weights
    #   cell:                       str, LSTM cell type
    #   features:                   int, number defining input slices

    weights = []
    bias = []
    for unit in unit_list:
        #print(connection, connections)
        
        # Define unit amounts
        in_w = features
        out_w = unit

        if transpose:
            bias_num = in_w
        else:
            bias_num = out_w
        
        # Create weights
        if cell == 'naive':
            w_var = {
                'input': {
                    'i': create_weights([in_w, out_w], dtype=weight_dtype),
                    'h': create_weights([out_w, out_w], dtype=weight_dtype)
                    },
                'forget': {
                    'i': create_weights([in_w, out_w], dtype=weight_dtype),
                    'h': create_weights([out_w, out_w], dtype=weight_dtype)
                    },
                'update': {
                    'i': create_weights([in_w, out_w], dtype=weight_dtype),
                    'h': create_weights([out_w, out_w], dtype=weight_dtype)
                    },
                'output': {
                    'i': create_weights([in_w, out_w], dtype=weight_dtype),
                    'h': create_weights([out_w, out_w], dtype=weight_dtype)
                    }
                }
        elif cell == 'optimized':
            w_var = {
                'ih': create_weights([in_w, out_w * 4], dtype=weight_dtype),
                'hh': create_weights([out_w, out_w * 4], dtype=weight_dtype),
                    }

        if cell == 'symmetric':
            w_var = {
                'ih': create_weights([in_w, out_w * 4], dtype=weight_dtype),
                'hh': create_weights([in_w, out_w * 4], dtype=weight_dtype),
                'symmetric': create_weights([out_w, in_w], dtype=weight_dtype)
                }

        if use_bias:
            if cell in ['optimized', 'symmetric']:
                # Create bias
                b_var = {'b': create_bias([bias_num * 4], dtype=weight_dtype)}
            elif cell == 'naive':
                b_var = {
                        'input': create_bias([bias_num], dtype=weight_dtype),
                        'forget': create_bias([bias_num], dtype=weight_dtype),
                        'update': create_bias([bias_num], dtype=weight_dtype),
                        'output': create_bias([bias_num], dtype=weight_dtype),
                        }

            if cell == 'symmetric':
                b_var['symmetric'] = create_bias([in_w], dtype=weight_dtype)

        else:
            b_var = None
    
        weights.append(w_var)
        bias.append(b_var)

        # In case cells are stacked along x
        features = unit
    
    return (weights, bias)

def LSTM_transpose_weights_creation_loop(weights_list, bias=None):

    weights = []
    biases = []
    for i, weight in enumerate(weights_list):
        for gate, ws in weight.items():
            for l, w in gate.items():
                w_var = tf.transpose(w)
        
        if bias is not None:
            b_var = bias[i]
        else:
            b_var = None
        
        weights.append(w_var)
        biases.append(b_var)

    return (weights, bias)

def initialize_LSTM_layer(
        layer_name, 
        input_dtype, 
        conf, 
        weights, 
        bias, 
        transpose
        ):
    # Create weights for the LSTM cell layer if not made
    # In:
    #   conf:                   dict, configuration
    # Out:
    #   (weigths, bias):        (dict, dict) modified weights dicts
    
    if not layer_name in list(weights.keys()):
        if isinstance(conf['units'], list):
            trainable_vars = True
            # Create new weights
            rws, rbs = LSTM_weights_creation_loop(
                        conf['units'], 
                        conf['use_bias'], 
                        input_dtype,
                        conf['cell'],
                        conf['features'],
                        transpose
                        )

        elif isinstance(conf['units'], str):
            trainable_vars = False
            # Configuration should habe the name of the layer whichs weights are used
            layer_to_reverse = conf['units']
        
            if conf['use_bias']:
                reversed_bs = list(reversed(bias[layer_to_reverse][1]))
            else:
                reversed_bs = None

            cws, cbs = LSTM_transpose_weights_creation_loop(
                list(reversed(weights[layer_to_reverse][1])),
                reversed_bs,
                )
        
        weights[layer_name] = (trainable_vars, rws)
        bias[layer_name] = (trainable_vars, rbs)

    return (weights, bias)

from .DENSE.layer import dense_layer
from .DENSE.builder import initialize_dense_layer

from .CONV.layer import conv_layer
from .CONV.builder import initialize_conv_layer

from .LSTM.layer import Naive_LSTM_cell, optimized_LSTM_cell, symmetric_LSTM_cell
from .LSTM.builder import initialize_LSTM_layer

from .. import tf, npprod

def get_numpy_shape(x):
    if hasattr(x, 'numpy'):
        return x.numpy().shape
    else:
        return x.shape


class Layer_Handler:

    def __init__(self):
        self.shapes = {}

    def init_shapes(self, layer_name):
        # Initializes the shapes dict

        if layer_name not in list(self.shapes.keys()):
            self.shapes[layer_name] = {}

    def check_transpose(self, conf):
        # Uses the conf dict to resolve if the layer is transpose layer or not
        
        def add_last_hidden_to_conf():
            # Add last layer to transpose layers
            if isinstance(conf['weights'], list):
                if conf['weights'][-1] != self.out_dense:
                    conf['weights'].append(int(self.out_dense))

        if 'transpose' in list(conf.keys()):
            if 'weights' in list(conf.keys()):
                add_last_hidden_to_conf()
            return conf['transpose']
        elif 'weights' in list(conf.keys()):
            if isinstance(conf['weights'], str):
                add_last_hidden_to_conf()
                return True
            else:
                return False
        else:
            return False

    def handle_dense_input_shape(self, x, conf):
        # Flats the input is not in shape (batch, features)
        # In:
        #   x:                      Tensor, input
        #   conf:                   dict, configuration
        # Out:
        #   x:                      Tensor, shaped or unshaped input
        #   original_shape:         tuple, the shape before flattening
        #   flat_shape:             tuple, shape after flattening

        if len(x.shape) > 2:
            original_shape = get_numpy_shape(x)
            flatted_shape = npprod(x.shape[1:])
            # Flat input if needed
            x = tf.reshape(x, (x.shape[0], flatted_shape))
            # Add flatted layer to weights
            self.original = original_shape
            self.out_dense = flatted_shape
            if conf['weights'][0] != int(flatted_shape):
                conf['weights'].insert(0, int(flatted_shape))
        else:
            original_shape = None
            flatted_shape = None
            if not isinstance(conf['weights'], str):
                if conf['weights'][0] != x.shape[1:][0]:
                    conf['weights'].insert(0, x.shape[1:][0])
        
        return (x, original_shape, flatted_shape)

    def handle_transpose_shape_fetch(self, layer, t_layer_name, w_num):
        
        # Define the correct shapes block
        t_block = self.shapes[t_layer_name]
        
        # Reverse the order of layers
        if not hasattr(self, 'transpose_order'):
            self.transpose_order = list(reversed(sorted(t_block.keys())))
        
        t_layer = self.transpose_order[layer]

        # Last layer takes the out_shape from Input shape
        # Every else takes it from output shape
        if layer != w_num:
            key = 'IN'
        else:
            key = 'OUT'
        
        out_shape = t_block[t_layer][key]
        
        return out_shape

    def Dense(  self,
                x, 
                weights, 
                bias, 
                conf, 
                layer_name, 
                input_dtype, 
                training=False
                ):

        # Create or load weights to a dense layer
        # In:
        #   x:                          Tensor, input
        #   weights:                    list, of tensorflow weights
        #   conf:                       dict, configurations
        #   layer_name:                 str, name for the layer
        #   input_dtype:                str, datatype for weights
        # Out:
        #   x:                          Tensor, output from the layers
        
        # Initialize the shapes object
        self.init_shapes(layer_name)
        
        # Check if the layer is a transposed layer
        transpose = self.check_transpose(conf)

        # If input is not in shape (batch, features) flat the inputs
        x, original_shape, flatted_shape = self.handle_dense_input_shape(x, conf)
        
        # Initialize layers
        weights, bias = initialize_dense_layer(layer_name, input_dtype, conf, weights, bias, transpose)
        
        # Choose weights
        ws = weights[layer_name][1]
        bs = bias[layer_name][1]
        shapes = self.shapes[layer_name]
        # Feed the input through the dense layer
        for layer, w in enumerate(ws):

            if layer not in shapes.keys():
                shapes[layer] = {'IN':get_numpy_shape(x)}
            else:
                shapes[layer]['IN'] = get_numpy_shape(x)
            
            x = dense_layer(
                x, 
                w, 
                bs[layer],
                conf['activations'][layer], 
                conf['dropouts'][layer], 
                training,
                transpose
                )
        
            shapes[layer]['OUT'] = get_numpy_shape(x)
                                
        return x

    def Convo(self,
                x, 
                weights, 
                bias, 
                conf, 
                layer_name, 
                input_dtype, 
                training=False,
                ):

        # Create or load weights to a dense layer
        # In:
        #   x:                          Tensor, input
        #   weights:                    list, of tensorflow weights
        #   conf:                       dict, configurations
        #   layer_name:                 str, name for the layer
        #   input_dtype:                str, datatype for weights
        # Out:
        #   x:                          Tensor, output from the layers

        
        transpose = self.check_transpose(conf)
        
        if len(x.shape) < 4:
            if hasattr(self, 'original'):
                x = tf.reshape(x, self.original)
        
        # Initialize layers
        weights, bias = initialize_conv_layer(layer_name, input_dtype, conf, weights, bias, transpose)
        
        # Initialize the shapes object
        self.init_shapes(layer_name)
        
        # Choose weights
        ws = weights[layer_name][1]
        bs = bias[layer_name][1]
        shapes = self.shapes[layer_name]
        
        # Feed the input through the convolutional layer
        for layer, w in enumerate(ws):
            
            if layer not in shapes.keys():
                shapes[layer] = {'IN':get_numpy_shape(x)}
            else:
                shapes[layer]['IN'] = get_numpy_shape(x)
            
            if transpose:
                if 'transpose' in list(conf.keys()):
                    if isinstance(conf['transpose'], str):
                        out_shape = self.handle_transpose_shape_fetch(
                                            layer, 
                                            conf['transpose'], 
                                            len(ws)
                                            )
                    else: # isinstance(transpose, bool):
                        out_shape = self.original

                else:
                    print("In convolutional transpose layers you must have 'transpose' key and its value is the layer name of which is to be transposed...")
                    exit()
            else:
                out_shape = False
            
            x = conv_layer(
                        x, 
                        w,
                        conf['strides'][layer],
                        conf['paddings'][layer],
                        conf['poolings'][layer],
                        bs[layer],
                        conf['activations'][layer], 
                        conf['batch_norms'][layer],
                        conf['dropouts'][layer], 
                        training,
                        out_shape
                        )

            shapes[layer]['OUT'] = get_numpy_shape(x)
        
        return x

    def LSTM(self,
                x, 
                weights, 
                bias, 
                conf, 
                layer_name, 
                input_dtype, 
                training=False,
                initial_state=None,
                init_output=True,
                ):

        # Create or load weights to a dense layer
        # In:
        #   x:                          Tensor, input
        #   weights:                    list, of tensorflow weights
        #   conf:                       dict, configurations
        #   layer_name:                 str, name for the layer
        #   input_dtype:                str, datatype for weights
        #   training:                   bool, defines if model is in training
        #   initial_state:              Tensor, state tensor for cell
        #   init_output:                bool, if cell output is initialized
        # Out:
        #   x:                          Tensor, output from the layers
    
        # Define cell
        if conf['cell'] == 'optimized':
            cell = optimized_LSTM_cell
            init_shapes = [[x,shape[0], conf['units'][0]], [x,shape[0], conf['units'][0]]]
        elif conf['cell'] == 'naive':
            cell = Naive_LSTM_cell
            init_shapes = [[x,shape[0], conf['units'][0]], [x,shape[0], conf['features']]]
        elif conf['cell'] == 'symmetric':
            cell = symmetric_LSTM_cell
            init_shapes = [[x,shape[0], conf['units'][0]], [x,shape[0], conf['features']]]
        
        # Remove last dim if it is 1
        #if x.shape[-1] == 1:
        #    x = tf.reshape(x, [s for s in x.shape[:-1]])
        
        # What happends inside the cell
        def loop(last_output, last_cell_state, step):
            print("Input shape:")
            if init_output:
                if len(x.shape) == 3:
                    inp = tf.slice(x, [0, 0, step], [-1, -1, 1])
                    inp = tf.reshape(inp, (inp.shape[0], inp.shape[1]))
                elif len(x.shape) == 4:
                    inp = tf.slice(x, [0, 0, step, 0], [-1, -1, 1, -1])
                    dim = tf.reduce_prod(inp.shape[1:])
                    inp = tf.reshape(inp, [-1, dim])
            else:
                inp = last_output
            output, state = cell(
                    inp,
                    (last_output, last_cell_state), 
                    w,
                    bs[layer],
                    transpose
                    )
            all_outputs.write(step, output)
            all_states.write(step, state)

            return (output, state, tf.add(step, 1))
        
        # Initialize the shapes object
        self.init_shapes(layer_name)
        
        # Check if the layer is a transposed layer
        transpose = self.check_transpose(conf)

        # Initialize layers
        weights, bias = initialize_LSTM_layer(layer_name, input_dtype, conf, weights, bias, transpose)
 
        # Initialize placeholders
        all_outputs = tf.TensorArray(input_dtype, size=conf['sequence'])
        all_states = tf.TensorArray(input_dtype, size=conf['sequence'])
        
        # Initialize cell
        if initial_state is None:
            initial_state = tf.zeros(init_shapes[0], dtype=input_dtype)
        if init_output:
            initial_output = tf.zeros(init_shapes[1], dtype=input_dtype)
        else:
            initial_output = x
        print("INITS")
        print(initial_state.shape)
        print(initial_output.shape)
        # Choose weights
        ws = weights[layer_name][1]
        bs = bias[layer_name][1]
        shapes = self.shapes[layer_name]
        
        # tf.while statement
        for_each_timestep = lambda a, b, step: tf.less(step, conf['sequence'])

        # Feed the input through the dense layer
        for layer, w in enumerate(ws):

            if layer not in shapes.keys():
                shapes[layer] = {'IN':get_numpy_shape(x)}
            else:
                shapes[layer]['IN'] = get_numpy_shape(x)
            
            final_output, final_state, _ = tf.while_loop(
                                            for_each_timestep, 
                                            loop,
                                            (initial_output, initial_state, 0),
                                            parallel_iterations=conf['parallel_iters']
                                            )

            shapes[layer]['OUT'] = get_numpy_shape(final_output)

        if conf['stack']:
            x = tf.stack(all_outputs)
        else:
            x = final_output
                                
        return (x, final_state)

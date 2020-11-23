from . import configurations

from .. import tfdata, tfoptimizers, tfreshape, exit, Path, optimization, loss_functions, test_functions, Layer_Handler, tf_training_loop, save_configuration, save_weights, load_weights, load_configuration, handle_init, npprod

class Model:

    def __init__(self, conf_name):
        
        # Initialize weight and bias
        self.weights = {}
        self.bias = {}
        self.layer_handler = Layer_Handler()
        # Call initialization handler
        handle_init(self, conf_name, configurations)

    def save(self):
        # Saves model weight variables
        
        w = self.weights
        b = self.bias

        path = Path('models/NeuralNetworks/saved_models/')
        path = save_weights(w, b, path.joinpath(self.conf_name))
        save_configuration(self.c, self.conf_name, path)
    
    def load(self, path):
        
        #Load weight variables
        weights, biases = load_weights(path)
        #print(type(weights))
        self.weights = {}
        self.bias = {}
        for layer_name in weights.item():
            self.weights[layer_name] = weights.item().get(layer_name)
            self.bias[layer_name] = biases.item().get(layer_name)

    def train(self,
            datasets,
            batch_size, 
            epochs, 
            learning_rate,
            loss_function='cross_entropy_w_sigmoid', 
            optimization_function='classifier',
            debug=False
            ):
        
        if isinstance(datasets, tuple):
            train, validate = datasets
        else:
            train = datasets
            validate = None

        #Define optimizer
        optimizer = tfoptimizers.Adam(learning_rate)
        #Define loss function
        loss_function = getattr(loss_functions, loss_function)
        #Define optimization function
        if 'autoencoder' == optimization_function:
            optimization_function = 'classifier'
            autoencoder = True
        else:
            autoencoder = False
        opt = getattr(optimization, optimization_function)
        
        #Dataset operations
        #Cache
        #dataset.cache()
        #Batch
        if batch_size != 0:
            train = train.batch(batch_size, drop_remainder=False)
        else:
            train = train.batch(1)

        #Start training
        tf_training_loop(
                train,
                validate,
                self, 
                loss_function, 
                opt, 
                optimizer, 
                epochs, 
                True, 
                autoencoder=autoencoder,
                debug=False
                )
        
        if not debug:
            self.save()
        print("Training finished...")

    def handle_layers(self, x, config, name_specifier='', training=False):
        # Clear specifier if the layer is on the main configuration (not encoder or decoder)
        if name_specifier == 'main':
            name_specifier = ''

        for i, (layer_type, conf) in enumerate(config.items()):
            
            layer_name = name_specifier+'_'+layer_type+'_'+str(i)
            inputs = [x, self.weights, self.bias, conf, layer_name, self.c['data_type'], training]
            if layer_type in dir(self.layer_handler):
                x = getattr(self.layer_handler, layer_type)(*inputs)
            else:
                print("Layer type: ", layer_type, " was not found...")
                exit()
        
        return x

    def encoder(self, x, training=False):
        if 'encoder'in list(self.c.keys()):
            handle_layers(
                    x,
                    self.c['encoder']['layers'],
                    'encoder',
                    training
                    )
        else:
            print("Model has not a defined encoder part...")
            exit()

    def decoder(self, x, training=False):
        if 'decoder' in list(self.c.keys()):
            handle_layers(
                    x,
                    self.c['decoder']['layers'],
                    'decoder',
                    training
                    )
        else:
            print("Model has not a defined decoder part...")
            exit()

    def run(self, x, training=False):
        
        fed_input_shape = x.shape
        
        # Reshape output
        if x.shape[1:] != self.c['input_shape']:
            in_shape = self.c['input_shape'].copy()
            in_shape.insert(0, -1)
            if npprod(x.shape[1:]) == npprod(in_shape[1:]):
                x = tfreshape(x, in_shape)
        
        # Get all layer configurations in a list
        if not hasattr(self, 'layer_confs'):
            layer_confs = {}
            for key, c in list(self.c.items()):
                
                if key == 'layers':
                    layer_confs['main'] = self.c['layers']
                elif isinstance(c, dict):
                    if 'layers' in list(c.keys()):
                        layer_confs[key] = c['layers']

            self.layer_confs = layer_confs
        
        # Handle layer building and feeding through the model
        for name, layer_conf in self.layer_confs.items():
            x = self.handle_layers(x, layer_conf, name, training)
        
        # Reshape output
        if x.shape != fed_input_shape:
            if npprod(x.shape[1:]) == npprod(fed_input_shape[1:]):
                x = tfreshape(x, fed_input_shape)
 
        return x

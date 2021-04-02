# RNN MNIST Autoencoder
conf = {
    'train_params':{
        'batch_size':1000, 
        'epochs':1, 
        'learning_rate':0.001, 
        'loss_function':'mean_squared_error',
        'optimization_function':'autoencoder'
        },
    
    'input_shape':[28, 28, 1],
    'data_type':'float32',
   
    'encoder':{
        'layers':{
            
            'LSTM':{
                'cell': 'optimized',
                'units': [1024],
                'features': 28,
                'sequence': 28,
                'use_bias': True,
                'stack': False,
                'parallel_iters': 32,
                'transpose': False
                },

            'Dense': {
                'weights':[1024, 10],
                'use_bias':True,
                'activations':['leaky_relu'],
                'dropouts':[None],
                'transpose': False
                }
            },
        },
    
    'decoder':{
        'layers':{
            
            'Dense': {
                'weights':[1024],
                'use_bias':True,
                'activations':['leaky_relu'],
                'dropouts':[None],
                'transpose': True
                },

            'LSTM':{
                'cell': 'optimized',
                'units': [1024],
                'features': 28,
                'sequence': 28,
                'use_bias': True,
                'stack': False,
                'parallel_iters': 32,
                'transpose':'encoder_LSTM_0',
                },

            },
        },
}

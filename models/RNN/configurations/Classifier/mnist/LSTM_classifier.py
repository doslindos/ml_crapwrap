# RNN MNIST Classifier
conf = {
    'train_params':{
        'batch_size':1000, 
        'epochs':1, 
        'learning_rate':0.0002, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    
    'input_shape':[28, 28, 1],
    'data_type':'float32',
    'output_shape':[10],
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
            'activations':['softmax'],
            'dropouts':[None]
            }
        },
}

# Dense autoencoder
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
    'output_shape':[1, 28, 28, 1],   
    'encoder':{
        'layers':{
            'Dense': {
                'weights':[512, 512, 10],
                'use_bias':True,
                'activations':['relu', 'relu', 'softmax'],
                'dropouts':[0.2, 0.2, None]
                },
            },
        },
    
    'decoder':{
        'layers':{
            'Dense': {
                'weights':[10, 512, 512],
                'use_bias':True,
                'activations':['relu', 'relu', 'relu'],
                'dropouts':[0.2, 0.2, None],
                'transpose':True
                },
            },
        },
}

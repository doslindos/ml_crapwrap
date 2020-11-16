# Models for Spotify features

conf = {
    'train_params':{
        'batch_size':100, 
        'epochs':1, 
        'learning_rate':0.0001, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    'input_shape':[10],
    'data_type':'float64',
    'layers':{
        'Dense': {
            'weights':[100, 100, 10],
            'use_bias':True,
            'activations':['relu', 'relu', 'softmax'],
            'dropouts':[0.2, 0.2, None]
        },
    },

}

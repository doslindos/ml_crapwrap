# Models for Spotify features

conf = {
    'train_params':{
        'batch_size':100, 
        'epochs':3, 
        'learning_rate':0.001, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    'input_shape':[12],
    'data_type':'float32',
    'output_shape':[10],
    'layers':{
        'Dense': {
            'weights':[100, 10],
            'use_bias':True,
            'activations':['relu', 'softmax'],
            'dropouts':[0.2, None]
        },
    },

}

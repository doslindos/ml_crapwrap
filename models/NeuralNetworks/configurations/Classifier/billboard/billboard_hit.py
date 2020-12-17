
# Models for Spotify features of billboard list songs

conf = {
    'train_params':{
        'batch_size':100, 
        'epochs':3, 
        'learning_rate':0.00001, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    'input_shape':[14],
    'data_type':'float32',
    'output_shape':[2],
    'layers':{
        'Dense': {
            'weights':[10, 2],
            'use_bias':True,
            'activations':['relu', 'softmax'],
            'dropouts':[None, None]
        },
    },

}

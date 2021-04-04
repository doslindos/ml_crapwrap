# Models for Spotify features of billboard list songs

conf = {
    'train_params':{
        'batch_size':100, 
        'epochs':3, 
        'learning_rate':0.00000001, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    'input_shape':[12],
    'data_type':'float32',
    'output_shape':[6],
    'layers':{
        'Dense': {
            'weights':[10, 6],
            'use_bias':True,
            'activations':['relu', 'softmax'],
            'dropouts':[None, None]
        },
    },

}

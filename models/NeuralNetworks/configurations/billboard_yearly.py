# Models for Spotify features of billboard list songs

conf = {
    'train_params':{
        'batch_size':10, 
        'epochs':10, 
        'learning_rate':0.1, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    'input_shape':[12],
    'data_type':'float32',
    'layers':{
        'Dense': {
            'weights':[100, 62],
            'use_bias':True,
            'activations':['sigmoid', None],
            'dropouts':[None, None]
        },
    },

}

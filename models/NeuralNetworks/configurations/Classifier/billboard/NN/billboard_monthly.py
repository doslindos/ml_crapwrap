
# Models for Spotify features of billboard list songs

conf = {
    'train_params':{
        'batch_size':100, 
        'epochs':5, 
        'learning_rate':0.001, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    'input_shape':[12],
    'data_type':'float32',
    'output_shape':[12],
    'layers':{
        'Dense': {
            'weights':[100, 12],
            'use_bias':True,
            'activations':['relu', None],
            'dropouts':[None, None]
        },
    },

}

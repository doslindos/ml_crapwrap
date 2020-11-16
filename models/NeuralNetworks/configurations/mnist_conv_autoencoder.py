# Convolutional-Dense MNIST autoencoder

conf = {
    
    'train_params':{
        'batch_size':100, 
        'epochs':5, 
        'learning_rate':0.001, 
        'loss_function':'mean_squared_error',
        'optimization_function':'autoencoder'
        },
    
    'input_shape':[28, 28, 1],
    'data_type':'float32',
    
    'encoder':{
        'layers':{
            
            'Convo':{
                'kernel_sizes':[[3, 3], [5, 5]],
                'filters':[1, 32, 64],
                'strides':[[1, 1], [2, 2]],
                'poolings':[None, None],
                'paddings':['SAME', 'SAME'],
                'batch_norms':[False, False],
                'use_bias':True,
                'activations':['leaky_relu', 'leaky_relu'],
                'dropouts':[None, None]
                },
    
            'Dense': {
                'weights':[100],
                'use_bias':True,
                'activations':[None],
                'dropouts':[None]
                },
            },
        },
    
    'decoder':{
        'layers':{
            
            'Dense': {
                'weights':[100],
                'use_bias':True,
                'activations':['leaky_relu'],
                'dropouts':[None],
                'transpose':True
                },
            
            'Convo':{
                'kernel_sizes':[[5, 5], [3, 3]],
                'filters':[64, 32, 1],
                'strides':[[2, 2], [1, 1]],
                'poolings':[None, None ],
                'paddings':['SAME', 'SAME'],
                'batch_norms':[False, False],
                'use_bias':True,
                'activations':['leaky_relu', 'leaky_relu'],
                'dropouts':[None, None],
                'transpose':'encoder_Convo_0',
                },
            },
        },

}

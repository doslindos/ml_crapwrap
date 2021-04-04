# Convolutional-Dense MNIST autoencoder

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
            
            'Convo':{
                'kernel_sizes':[[3, 3]],
                'filters':[1, 32],
                'strides':[[1, 1]],
                'poolings':[None],
                'paddings':['SAME'],
                'batch_norms':[False],
                'use_bias':True,
                'activations':['leaky_relu'],
                'dropouts':[None]
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
                'kernel_sizes':[[3, 3]],
                'filters':[32, 1],
                'strides':[[1, 1]],
                'poolings':[None ],
                'paddings':['SAME'],
                'batch_norms':[False],
                'use_bias':True,
                'activations':['leaky_relu'],
                'dropouts':[None],
                'transpose':'encoder_Convo_0',
                },
            },
        },

}

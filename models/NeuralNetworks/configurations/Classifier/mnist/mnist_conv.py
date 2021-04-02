# Convolutional MNIST Classifier
conf = {
    'train_params':{
        'batch_size':500, 
        'epochs':1, 
        'learning_rate':0.0015, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    
    'input_shape':[28, 28, 1],
    'data_type':'float32',
    'output_shape':[10],
    'layers':{
        
        'Convo':{
            'kernel_sizes':[[5, 5]],
            'filters':[1, 32],
            'strides':[[2, 2]],
            'poolings':[[2, 2]],
            'paddings':['SAME'],
            'batch_norms':[False],
            'use_bias':True,
            'activations':['leaky_relu'],
            'dropouts':[None]
            },
    
        'Dense': {
            'weights':[100, 10],
            'use_bias':True,
            'activations':['leaky_relu', 'softmax'],
            'dropouts':[None, None]
            },
        },
}

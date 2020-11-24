# Convolutional MNIST Classifier
conf = {
    'train_params':{
        'batch_size':100, 
        'epochs':3, 
        'learning_rate':0.0001, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    
    'input_shape':[28, 28, 1],
    'data_type':'float32',
    'layers':{
        
        'Convo':{
            'kernel_sizes':[[3, 3]],
            'filters':[1, 28],
            'strides':[[1, 1]],
            'poolings':[[2, 2]],
            'paddings':['SAME'],
            'batch_norms':[False],
            'use_bias':True,
            'activations':['relu'],
            'dropouts':[None]
            },
    
        'Dense': {
            'weights':[512, 10],
            'use_bias':True,
            'activations':['relu', 'softmax'],
            'dropouts':[0.3, None]
            },
        },
}

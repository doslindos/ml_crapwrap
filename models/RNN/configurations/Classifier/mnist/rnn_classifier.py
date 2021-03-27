# RNN MNIST Classifier
conf = {
    'train_params':{
        'batch_size':1000, 
        'epochs':1, 
        'learning_rate':0.0002, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    
    'input_shape':[28, 28, 1],
    'data_type':'float32',
    'output_shape':[10],
    'layers':{
        
        'RNN':{
            'kernel_sizes':[[3, 3]],
            'filters':[1, 32],
            'strides':[[1, 1]],
            'poolings':[[2, 2]],
            'paddings':['SAME'],
            'batch_norms':[False],
            'use_bias':True,
            'activations':['relu'],
            'dropouts':[None]
            },
        },
}

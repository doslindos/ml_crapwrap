conf = {
    # Models for MNIST Images in Grayscale colorschema
    # Dense classifier
    'train_params':{
        'batch_size':100, 
        'epochs':3, 
        'learning_rate':0.0001, 
        'loss_function':'cross_entropy',
        'optimization_function':'classifier'
        },
    'input_shape':[28, 28, 1],
    'data_type':'float32',
    'output_shape':[10],
    'layers':{
        'Dense': {
            'weights':[512, 512, 10],
            'use_bias':True,
            'activations':['relu', 'relu', 'softmax'],
            'dropouts':[0.2, 0.2, None]
            },
        }
    
}

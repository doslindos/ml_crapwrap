from sklearn.neural_network import MLPClassifier


conf = {
        'model': MLPClassifier(),
        'params':[{
            'hidden_layer_sizes':[
                    (5,), 
                ], 
            'activation':[
                'relu'
                ], 
            'solver':[
                'adam',
                ],
            'learning_rate_init':[
                0.001,
                ],
            'max_iter':[
                1000,
                ]
            }],
        'name':'MLPClassifier'
}

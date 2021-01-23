from sklearn.neural_network import MLPClassifier


conf = {
        'model': MLPClassifier(),
        'params':[{
            'hidden_layer_sizes':[
                    (5,), 
                    (10,), 
                    (25,), 
                    (50,), 
                    (100,),
                    (5,5), 
                    (10,10), 
                    (25,25), 
                    (50,50), 
                    (100,100)
                ], 
            'activation':[
  #              'identity', 
  #              'logistic', 
                'tanh', 
                'relu'
                ], 
            'solver':[
                'adam',
                ],
            'learning_rate_init':[
                0.01,
                0.001,
                0.0001,
                0.00001
                ],
            'max_iter':[
                500,
                1000,
                2000
                ]
            }],
        'name':'MLPClassifier'
}

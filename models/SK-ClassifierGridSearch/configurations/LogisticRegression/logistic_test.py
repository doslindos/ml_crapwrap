from sklearn.linear_model import LogisticRegression


conf = {
        'model': LogisticRegression(),
        'params':[{
            'penalty':[
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

from sklearn.linear_model import LogisticRegression


conf = {
        'model': LogisticRegression(),
        'params':[{
            'penalty':[
                    'l2', 
                ], 
            'dual':[
                False
                ], 
            'tol':[
                1e-4,
                ],
            'C':[
                1.0,
                ],
            'n_jobs':[
                -2,
                ],
            'max_iter':[
                100
                ]
            }],
        'name':'LogisticReg'
}

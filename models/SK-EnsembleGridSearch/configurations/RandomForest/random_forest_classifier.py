conf = {
        'models':[
            {
                'module': 'sklearn.ensemble',
                'search_params':[
                    {'n_estimators':[5, 10, 25, 50, 100], 'warm_start':[True, False], 'max_depth':[None, 10, 50]}
                    ],
                'model':'RandomForestClassifier'
            },
            ]

}

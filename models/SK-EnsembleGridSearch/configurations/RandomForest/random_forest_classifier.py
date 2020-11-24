from sklearn.ensemble import RandomForestClassifier

conf = {
        'models':[
            {
                'model': RandomForestClassifier(),
                'params':[
                    {'n_estimators':[50, 100, 150, 250], 'warm_start':[True, False], 'max_depth':[None, 10, 50]}
                    ],
                'name':'RandomForestClassifier'
            },
            ]

}

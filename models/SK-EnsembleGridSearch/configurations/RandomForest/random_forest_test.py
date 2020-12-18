from sklearn.ensemble import RandomForestClassifier

conf = {
        'models':[
            {
                'model': RandomForestClassifier(),
                'params':[
                    {
                        'n_estimators':[100], 
                        'warm_start':[False], 
                        'max_depth':[None]}
                    ],
                'name':'RandomForestClassifier'
            },
            ]

}

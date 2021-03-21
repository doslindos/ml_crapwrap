conf = {
        'models':[
            {
                'module': 'sklearn.ensemble',
                'search_params':[
                    {
                        'n_estimators':[100], 
                        'warm_start':[False], 
                        'max_depth':[None]}
                    ],
                'model':'RandomForestClassifier'
            },
            ]

}

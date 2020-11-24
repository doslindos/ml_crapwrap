from sklearn.ensemble import GradientBoostingClassifier

conf = {
    'models':[
        {
        'model': GradientBoostingClassifier(),
        'params':[
                {
                'n_estimators':[50, 100, 150, 200], 
                'warm_start':[True, False], 
                'max_depth':[3, 5, 10]
                },  
                ],
        'name':'GradientBoost'
        },
    ]

}

conf = {
    'models':[
        {
        'module': 'sklearn.ensemble',
        'search_params':[
            {
        'n_estimators':[50, 100, 200, 250], 
        'warm_start':[True, False], 
        'max_depth':[None, 10, 50]
            }  
        ],
        'model':'ExtraTreesClassifier'
            },
]}

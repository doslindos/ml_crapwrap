from sklearn.ensemble import ExtraTreesClassifier,

conf = {
    'models':[
        {
        'model': ExtraTreesClassifier(),
        'params':[
            {
        'n_estimators':[50, 100, 200, 250], 
        'warm_start':[True, False], 
        'max_depth':[None, 10, 50]
            }  
        ],
        'name':'ExtraTreesClassifier'
            },
]}

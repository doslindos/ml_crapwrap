conf = {

'models':[
    {
    'module': 'sklearn.ensemble',
    'model': 'BaggingClassifier', 
        'search_params':[
            {
            'base_estimator':[
                {'module': 'sklearn.neighbors', 'model': 'KNeighborsClassifier', 'params': {}},
                {'module': 'sklearn.tree', 'model': 'DecisionTreeClassifier', 'params': {}},
                {'module': 'sklearn.svm', 'model': 'SVC', 'params': {}}
                ],
            'n_estimators':[100, 200, 300],
            'warm_start':[True, False]
            },
        ],
    },
]
}

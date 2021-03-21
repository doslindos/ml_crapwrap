conf = {
'models':[
    {
    'module': 'sklearn.ensemble',
    'params': {'estimators': [
        {'name': 'KN', 'module': 'sklearn.neighbors', 'model': 'KNeighborsClassifier', 'params': {}},
        {'name': 'DT', 'module': 'sklearn.tree', 'model': 'DecisionTreeClassifier', 'params': {}},
        {'name': 'SVC', 'module': 'sklearn.svm', 'model': 'SVC', 'params': {'probability': True}},
            ]
        },
    'search_params':[
                {
            'voting':['hard', 'soft']
                },  
            ],
    'model':'VotingClassifier'
            },

    ]
}

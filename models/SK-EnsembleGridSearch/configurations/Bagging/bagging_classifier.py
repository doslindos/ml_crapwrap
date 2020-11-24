from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  BaggingClassifier

conf = {

'models':[
    {
    'model': BaggingClassifier(), 
        'params':[
            {
            'base_estimator':[
                KNeighborsClassifier(), 
                DecisionTreeClassifier(), 
                SVC(), 
                ],
            'n_estimators':[100, 200, 300],
            'warm_start':[True, False]
            },
        ],
    'name':'Bagging(KNC,DTC,SVC)'
    },
]
}

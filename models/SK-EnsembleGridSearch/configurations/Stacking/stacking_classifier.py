from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

conf = {

'models':[
    {
    'model': StackingClassifier(
                estimators=[
                    ('KN', KNeighborsClassifier()), 
                    ('DT', DecisionTreeClassifier()), 
                    ('SVC', SVC(probability=True))]),
                'params':[
                    {}
                ],
                'name':'Stacking'
            },

                ]


}

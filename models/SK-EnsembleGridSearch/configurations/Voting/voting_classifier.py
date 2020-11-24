from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

conf = {
    'models':[
        {
        'model': VotingClassifier(
                        estimators=[
                            ('KN',KNeighborsClassifier()), 
                            ('DT', DecisionTreeClassifier()), 
                            ('SVC',SVC(probability=True))
                            ]
                        ),
        'params':[
            {
            'voting':['hard', 'soft']
            },  
            ],
        'name':'Voting(KNC,DTC,SVC)'
            },

    ]
}

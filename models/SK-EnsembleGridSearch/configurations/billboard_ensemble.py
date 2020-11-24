from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, BaggingRegressor, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

conf = {

'models':[
                {
                'model': BaggingClassifier(), 
                'params':[
                    {'base_estimator':[KNeighborsClassifier(), DecisionTreeClassifier(), SVC(), ],
                        'n_estimators':[100, 200, 300],
                        'warm_start':[True, False]
                        },
                ],
                'name':'Bagging(KNC,DTC,SVC)'
            },
                {
                'model': RandomForestClassifier(),
                'params':[
                    {'n_estimators':[50, 100, 150, 250], 'warm_start':[True, False], 'max_depth':[None, 10, 50]}
                    ],
                'name':'Random Forest'
            },
                {
                'model': ExtraTreesClassifier(),
                'params':[
                    {'n_estimators':[50, 100, 200, 250], 'warm_start':[True, False], 'max_depth':[None, 10, 50]}  
                ],
                'name':'Extra Trees'
            },
                {
                'model': GradientBoostingClassifier(),
                'params':[
                    {'n_estimators':[50, 100, 150, 200], 'warm_start':[True, False], 'max_depth':[3, 5, 10]},  
                ],
                'name':'GradientBoost'
            },
                {
                'model': HistGradientBoostingClassifier(),
                'params':[
                    {'max_depth':[None, 10, 15], 'l2_regularization':[0, 0.5], 'warm_start':[True, False]},  
                ],
                'name':'HistGradientBoost'
            },
                {
                    'model': VotingClassifier(estimators=[('KN',KNeighborsClassifier()), ('DT', DecisionTreeClassifier()), ('SVC',SVC(probability=True))]),
                'params':[
                    {'voting':['hard', 'soft']
                        },  
                ],
                'name':'Voting(KNC,DTC,SVC)'
            },
                {
                'model': StackingClassifier(estimators=[('KN', KNeighborsClassifier()), ('DT', DecisionTreeClassifier()), ('SVC', SVC(probability=True))]),
                'params':[
                    {}
                ],
                'name':'Stacking'
            },

                ]


}

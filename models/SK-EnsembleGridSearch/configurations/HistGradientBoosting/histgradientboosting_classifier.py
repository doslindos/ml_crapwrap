from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

conf = {

'models':[
        {
        'model': HistGradientBoostingClassifier(),
        'params':[
            {
            'max_depth':[None, 10, 15], 
            'l2_regularization':[0, 0.5], 
            'warm_start':[True, False]
            },  
                ],
        'name':'HistGradientBoost'
            },

    ]

}

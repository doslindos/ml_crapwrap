from . import skd, metrics, train_test_split, shuffle, nparray, npappend, results_to_nplist

def split_dataset(data, labels, train_size, make_shuffle=True):
    # Splits dataset into train and test set
    # In:
    #   data:                       numpy matrix?
    #   labels:                     numpy matrix?
    #   train_size:                 float, percentage of split between train and test
    #   make_shuffle:                    bool, True = shuffles data and labels
    # Out:
    #   tuple:                      (train data, test data, train labels, test labels)
    
    if make_shuffle:
        #Shuffle the dataset
        data, labels = shuffle(data, labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, stratify=labels)
    
    return (x_train, x_test, y_train, y_test)

def apply_dim_reduction(data, function='PCA'):
    # Takes results dict and returns reduced values
    # In:
    #   data:                   array, of data instances
    #   function:               str, sklearn decomposition function name
    # Out:
    #   (init, fit, transform)  tuple, (initialized sklearn object, fitted sklearn object, transformed values)


    #print(data.shape)
    init = getattr(skd, function)()
    fit = init.fit(data)
    transform = nparray(fit.transform(data))
    return (init, fit, transform)

def make_confusion_matrix(y_true, y_prediction, labels):
    # Displays multulabel confusion matrix
    # In:
    #   y_true:                 list, list of true values
    #   y_prediction:           list, list of predicted values
    #   labels:                 list, list of possible labels
    
    cm = metrics.confusion_matrix(y_true, y_prediction, labels=labels)
    print(cm)
    return cm

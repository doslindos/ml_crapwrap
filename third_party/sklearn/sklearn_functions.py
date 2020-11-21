from . import skd, metrics, train_test_split, shuffle, nparray, npappend, results_to_nplist, preprocessing

def split_dataset(data, labels, test_size, make_shuffle=True, validation=None):
    # Splits dataset into train and test set
    # In:
    #   data:                       numpy matrix?
    #   labels:                     numpy matrix?
    #   test_size:                 float, percentage of split between train and test
    #   make_shuffle:               bool, True = shuffles data and labels
    #   validation:                 float, if None validation set is not created
    # Out:
    #   tuple:                      (train data, test data, train labels, test labels)
    
    if make_shuffle:
        #Shuffle the dataset
        data, labels = shuffle(data, labels)
   
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=train_size, stratify=labels)
    

    if validation is not None:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation, stratify=y_train)
        return ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    
    return ((x_train, y_train),  (x_test, y_test))

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

def label_encoding(labels):
    # Uses preprocess.LabelEncoder to encode labels
    # In:
    #   labels:                 list, every instance is a label
    # Out:
    #   label_encoder:          Sklearn LabelEncoder object
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder

def one_hot_encoding(labels):
    # Uses preprocess.LabelEncoder to encode labels
    # In:
    #   labels:                 list, every instance is a label
    # Out:
    #   label_encoder:          Sklearn LabelEncoder object
    
    one_hot_encoder = preprocessing.OneHotEncoder()
    one_hot_encoder.fit(labels)
    return one_hot_encoder



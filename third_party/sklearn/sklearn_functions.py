import sklearn.decomposition as skd
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.utils import shuffle
from numpy import append as npappend, array as nparray, linspace as nplinspace
import matplotlib.pyplot as plt

from utils.utils import results_to_nplist

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
   
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify=labels)
    

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

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=nplinspace(.1, 1.0, 10)):
    # From https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

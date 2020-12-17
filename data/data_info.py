from plotting.util.plotting import build_histogram
from collections import Counter
from third_party.scipy.util import print_description

def frequencies(dataset):
    # Plots frequencies of a dataset by label
    
    # Get data and labels in a numpy array
    for batch in dataset.batch(dataset.cardinality()):
        labels = batch[1].numpy()
    
    # Convert labels to strings
    labels = [str(i) for i in labels]
    print(Counter(labels).most_common())
    # Get number of instances per label
    labels, amounts = zip(*sorted(Counter(labels).items(), key=lambda x: x[0]))

    build_histogram(amounts, labels)

def frequencies_by_feature(dataset):
    # Plots frequencies of a dataset by feature
    
    # Get data and labels in a numpy array
    for batch in dataset.batch(dataset.cardinality()):
        x = batch[0].numpy()
        print(x.shape)
        print("This dataset has ", x[0].shape, " features...")
        if len(x[0].shape) == 1:
            f = int(input("Choose feature (num): "))
            labels = x[:, f-1:f].flatten()
            print(labels.shape)
        else:
            print("More than 1 dim unimplemented...")
            exit()

        #labels = batch[1].numpy()
    
    print(Counter(labels).most_common())
    # Get number of instances per label
    labels, amounts = zip(*sorted(Counter(labels).items(), key=lambda x: x[0]))

    build_histogram(amounts, labels)

def describe(dataset):
    # Prints description of the data

    for batch in dataset.batch(dataset.cardinality()):
        data = batch[0].numpy()
        labels = batch[1].numpy()
    
    print("\nData description:")
    print_description(data)
    print("\nLabels description:")
    print_description(labels)

from plotting.util.plotting import build_histogram

def frequencies(dataset, label_key):
    # Plots frequencies of a dataset by label

    print(dataset, type(dataset))
    hist_data = dataset.label_counts(label_key)
    if len(hist_data) > 100:
        print("Unique sample count ", len(hist_data))
        print("Showing only values with value more than 1...")
        hist_data = {key:value for key, value in hist_data.items() if value > 1}
    labels, data = zip(*hist_data.items())
    build_histogram(data, labels)

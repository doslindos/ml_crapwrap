from . import nparray, npwhere, npfull, plt, get_cmap

def scatter(labels, data):
    # Create and show scatter plot
    # In:
    #   labels:                     array, label for x[i] and y[i]
    #   data:                       array, contains x, y and possible z lists
    
    fig = plt.figure()
    
    #2d plot setup
    if len(data) == 2:
        x, y = data
        z = None
        x = nparray(x)
        y = nparray(y)
        ax = plt.axes()

    #3d plot setup
    elif len(data) == 3:
        x, y, z = data
        x = nparray(x)
        y = nparray(y)
        z = nparray(z)
        ax = plt.axes(projection='3d')
    
    if labels is not None:
        #Take set of labels for colormap
        label_set = set(labels)
        colormap = get_cmap(len(label_set))
        labels = nparray(labels)

        #Loop through every unique value in labels and plot its values
        for i, label in enumerate(label_set):
            #Search indexes with label value
            indexes = npwhere(labels == label)[0]

            #Handle label colors
            c = colormap(labels[indexes])
            
            #Plot the data
            if z is None:
                ax.scatter(x[indexes], y[indexes], c=c, label=label)
            else:
                ax.scatter(x[indexes], y[indexes], z[indexes], c=c, label=label)
    else:
        #Plot the data without labels
        if z is None:
            ax.scatter(x, y)
        else:
            ax.scatter(x, y, z)
    
    #Show label colors and values
    plt.legend()
    #Show the plot
    plt.show()

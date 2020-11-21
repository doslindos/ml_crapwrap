from .. import apply_dim_reduction, results_to_nplist, exit, plt, nparray, npsum, npappend

def get_cmap(n, name='rainbow'):
    # Creates a plt colormap object which returns RGB values for the value
    # Is initialized with a colormap name and a number of values for which the colormap is divided for
    # In:
    #   n:                   int, number of possible colors
    #   name:                plt colormap name

    return plt.cm.get_cmap(name, n)

def format_data_to_plot(results, dims, function='PCA'):
    # Formats the data to be used by the plot
    # In:
    #   results:                    dict, keys = labels and values = model outputs
    #   dims:                       int, number of dimensions
    #   function:                   str, name of the dimension reduction function (sklearn.decomposition)
    # Out:
    #   (labels, data_lists)        tuple, (label array, x, y and possible z array)

    #Order results by labels
    results = dict(sorted(results.items(), key=lambda kv: kv[0]))
    
    # Results dict into data and lables arrays where label of data[i] is label[i]
    data, labels = results_to_nplist(results)
    instance_dims = data.shape[-1]
    
    #Applies dimension reduction if needed
    print("Data: ", data.shape)
    if instance_dims > dims and function is not None:
        print("Applying dimension reduction...")
        init, fit, data = apply_dim_reduction(data, function)
        #print(sum(fit.explained_variance_), fit.explained_variance_ratio_)
        print("Dim reduction done...")
    elif instance_dims < dims:
        print("Not enough values")
        exit()

    return (labels, list(zip(*data[:, :dims])))

def add_totals(confusion_matrix):
    cm = nparray(confusion_matrix)
    column_sum = npsum(cm, axis=0)
    row_sum = npsum(cm, axis=1)
    #print("COLUMN: ",column_sum)
    #print("ROW: ", row_sum)
    return (column_sum, row_sum)

def display_confusion_matrix(confusion_matrix, labels):
    
    #X and Y axis labels
    xlbl = 'Confusion matrix'
    ylbl = 'Actual'

    #Set x axis tick labels at the top
    plt.rcParams['xtick.bottom'] = False 
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = True
    
    #Get total amounts
    column_sum, row_sum = add_totals(confusion_matrix)

    #Create figure, axis and image objects
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)
    
    #Set labels
    ax.set_xticks(labels)
    #labelpad moves label down
    ax.set_xlabel(xlbl, labelpad=20)
    ax.set_yticks(labels)
    ax.set_ylabel(ylbl)
    
    # use for text
    #ax.set_xticklabels(labels)
    #ax.set_yticklabels(labels)
    
    #Loop dimensions and set values to in the image
    c = 'b'
    for i in range(len(labels)+1):
        for j in range(len(labels)+1):
            #Change exception into if
            try:
                ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='w')
            except:
                if c == 'b':
                    c = 'r'
                else:
                    c = 'b'

                if j == len(labels) and i < j:
                    inputs = column_sum[i]
                elif i == len(labels) and j < len(labels):
                    inputs = row_sum[j]
                else:
                    inputs = 'totals'
                ax.text(i, j, inputs, ha='center', va='center', color=c)
    
    #Top label
    ax.set_title("Predicted")
    
    #Create colorbar and set label
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.2)
    cbar.ax.set_ylabel("Number of predictions", rotation=-90, va='bottom')
    
    plt.show()

def build_histogram(data, labels):
    #print(labels, data)
    plt.bar(labels, data)
    plt.show()


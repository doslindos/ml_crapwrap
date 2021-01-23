from scipy.stats import describe
from numpy import set_printoptions, ndarray as ndarr
set_printoptions(suppress=True)

def print_description(x):
    desc_x = describe(x)
    
    if isinstance(x[0], ndarr):
    # Loop every "feature" and print its description
        for i in range(len(x[0])):
            f = i+1
            print("Feature %d "% f)
            print("Min: ", desc_x.minmax[0][i])
            print("Max: ", desc_x.minmax[1][i])
            print("Mean: ", desc_x.mean[i])
            print("Variance: ", desc_x.variance[i])
            print("Skewness: ", desc_x.skewness[i])
            print("Kurtosis: ", desc_x.kurtosis[i])
    else:

        print("Min: ", desc_x.minmax[0])
        print("Max: ", desc_x.minmax[1])
        print("Mean: ", desc_x.mean)
        print("Variance: ", desc_x.variance)
        print("Skewness: ", desc_x.skewness)
        print("Kurtosis: ", desc_x.kurtosis)

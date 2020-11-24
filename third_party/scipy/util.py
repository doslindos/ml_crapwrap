from scipy.stats import describe
from numpy import set_printoptions
set_printoptions(suppress=True)
def print_description(x):
    desc_x = describe(x)

    print("Min array: ", desc_x.minmax[0])
    print("Max array: ", desc_x.minmax[1])
    print("Mean array: ", desc_x.mean)
    print("Variance array: ", desc_x.variance)
    print("Skewness array: ", desc_x.skewness)
    print("Kurtosis array: ", desc_x.kurtosis)

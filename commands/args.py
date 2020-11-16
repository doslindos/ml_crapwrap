import argparse

def create_args(parser_args, add_args):
    #Creates Argument parser arguments
    # In:
    #   parser_args:                dict, Argument parser input
    #   add_args:                   dict, Argument parser add_argument method inputs
    # Out:
    #   parser.parse_args()         parsed arguments

    parser = argparse.ArgumentParser(**parser_args)
    for add in add_args:
        name = add.pop('name')
        parser.add_argument(*name, **add)
    return parser.parse_args()

def dataset():
    #Defines create function inputs and calls argument parser
    parser_args = {'description':'Create a dataset'}
    add_args = [
        {'name':['command'], 'type':str, 'help':'Main command'},
        {'name':['-s'], 'type':str, 'required':True, 'help':'Path to the script in data_script folder, to the excel file in data_file folder or name of the dataset (tensroflow-dataset "tfds_fetch")'},
        {'name':['-ff'], 'type':str, 'required':True, 'help':'Data fetch function'},
        {'name':['-cf'], 'type':str, 'default':None, 'help':'Create function'},
        {'name':['-name'], 'type':str, 'default':None, 'help':'Name of the dataset'},
        ]
    return create_args(parser_args, add_args)
 
def data():
    #Defines data information function inputs and calls argument parser
    parser_args = {'description':'Show data information'}
    add_args = [
        {'name':['command'], 'type':str, 'help':'Main command'},
        {'name':['-d'], 'type':str, 'required':True, 'help':'Name of the dataset'},
        {'name':['-info'], 'type':str, 'required':True, 'help':'Name of the information function'},
        {'name':['-l'], 'type':str, 'required':True, 'help':'Name of the data key'},
        {'name':['--merge_key'], 'type':str, 'default':None, 'help':'Name of the merge key'},
        ]
    return create_args(parser_args, add_args)

def train():
    #Defines training function inputs and calls argument parser
    parser_args = {'description':'Train a model'}
    add_args = [
        {'name':['command'], 'type':str, 'help':'Main command'},
        {'name':['-d'], 'type':str, 'required':True, 'help':'Name for the dataset'},
        {'name':['-pf'], 'type':str, 'default':None, 'help':'Name for the preprocessing function'},
        {'name':['-m'], 'type':str, 'default':"NeuralNetworks", 'help':'Model name'},
        {'name':['-c'], 'type':str, 'default':None, 'help':'Configurations name'},
        {'name':['--batch_size'], 'type':int, 'default':None, 'help':'Batch size, (do not change for sklearn functions)'},
        {'name':['--epochs'], 'type':int, 'default':None, 'help':'Number of times the whole dataset is trained to the model (do not define for sklearn functions)'},
        {'name':['--learning_rate'], 'type':float, 'default':None, 'help':'Learning rate (do not change for sklearn functions)'},
        {'name':['--loss_function'], 'type':str, 'default':None, 'help':'Learning rate (do not change for sklearn functions)'},
        {'name':['--optimization_function'], 'type':str, 'default':None, 'help':'Learning rate (do not change for sklearn functions)'},
        {'name':['--sub_sample'], 'type':int, 'default':None, 'help':'Use a subsample of the dataset'},
        ]
    return create_args(parser_args, add_args)
    
def test():
    #Defines test function inputs and calls argument parser
    parser_args = {'description':'Test a model'}
    add_args = [
        {'name':['command'], 'type':str, 'help':'Main command'},
        {'name':['-d'], 'type':str, 'required':True, 'help':'Dataset name'},
        {'name':['-pf'], 'type':str, 'default':None, 'help':'Name for the preprocessing function'},
        {'name':['-m'], 'type':str, 'default':"NeuralNetworks", 'help':'Model name'},
        {'name':['-test'], 'type':str, 'required':True, 'help':'Test type'},
        {'name':['-t'], 'type':int, 'default':None, 'help':'Test set size'},
        {'name':['--dataset_type'], 'type':str, 'default':'test', 'help':'Dataset type to be used'},
        {'name':['--sub_sample'], 'type':int, 'default':None, 'help':'Use a subsample of the dataset'}
        ]
    return create_args(parser_args, add_args)

def plot():
    #Defines plotting function inputs and calls argument parser
    parser_args = {'description':'Plot models outputs'}
    add_args = [
        {'name':['command'], 'type':str, 'help':'Main command'},
        {'name':['-d'], 'type':str, 'required':True, 'help':'Dataset name'},
        {'name':['-m'], 'type':str, 'default':"NN", 'help':'Model name'},
        {'name':['-plot'], 'type':str, 'required':True, 'help':'Plot type'},
        {'name':['--dataset_type'], 'type':str, 'default':'test', 'help':'Dataset type to be used'},
        {'name':['--plot_dims'], 'type':int, 'default':2, 'help':'Plot dimensions'},
        {'name':['--function'], 'type':str, 'default':'PCA', 'help':'Dimensionality reduction function. From sklearn.decomposition'},
        ]
    return create_args(parser_args, add_args)

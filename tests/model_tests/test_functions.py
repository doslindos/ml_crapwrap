from . import npeye, set_printoptions, nparray, cast, sklearn_functions, exit, display_confusion_matrix, ModelTester
from .util import parse_results

from cmd import Cmd

def classification_test(results, model, from_results=True):
    # Classification test
    # In:
    #   results:                dict, label - model output pairs
    #   model:                  model object, model used
    #   from_results:           bool, if true function uses results data else outputs from the model (not ready)

    predictions, y, labels = parse_results(results)
    print(nparray(predictions).shape, nparray(y).shape)
    accuracy(None, None, pred_y = (predictions, y))
    cm = sklearn_functions.make_confusion_matrix( y, predictions, labels)
    display_confusion_matrix(cm, labels)
    #print("Predictions: ", [npargmax(prediction) for prediction in predictions])
    #print("Labels: ", y.numpy())


def accuracy(results, model, pred_y=None):
    # Prints accuracy
    # In:
    #   results:                dict, label - model output pairs
    #   pred_y:                 tuple, allready parsed results
    
    if pred_y is None:
        if results is None:
            print("No inputs...")
            exit()

        pred_y = parse_results(results)

    print("Total accuracy: ", (nparray(pred_y[0]) == nparray(pred_y[1])).mean())

def testing_gui(model, results=None):
    # Opens testing GUI
    # In:
    #   results:                dict, label - model output pairs (Not used, just for quick fix)
    #   model:                  model object, model to be used
    
    ModelTester(model)

def play_with_model(model, results=None):
    
    class ModelPrompt(Cmd):
        
        def do_exit(self, inp):
            print("Hep")
            return True

        def do_feed_data_to_model(self, filepath):
            print(model)
            print(path)

    ModelPrompt().cmdloop()

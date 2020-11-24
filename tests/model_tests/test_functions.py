from . import npargmax, npeye, set_printoptions, nparray, cast, sklearn_functions, exit, display_confusion_matrix, ModelTester

def classification_test(results, model, from_results=True):
    # Classification test
    # In:
    #   results:                dict, label - model output pairs
    #   model:                  model object, model used
    #   from_results:           bool, if true function uses results data else outputs from the model (not ready)

    y = []
    predictions = []
    labels = len(results.keys())
    for key, result in results.items():
        for logits in result:
            y.append(key)
            if logits.shape[-1] > 2:
                predictions.append(npargmax(logits))
            else:
                predictions.append(logits)
            
    labels = [i for i in range(labels)]
    
    print(nparray(predictions).shape, nparray(y).shape)
    print("Total accuracy: ", (nparray(predictions) == nparray(y)).mean())
    cm = sklearn_functions.make_confusion_matrix( y, predictions, labels)
    display_confusion_matrix(cm, labels)
    #print("Predictions: ", [npargmax(prediction) for prediction in predictions])
    #print("Labels: ", y.numpy())


def accuracy(results, model, from_results=True):
    # Prints labeling accuracy
    # In:
    #   results:                dict, label - model output pairs
    #   model:                  model object, model used
    #   from_results:           bool, if true function uses results data else outputs from the model (not ready)
    
    for key, value in results.items():
        print("\nLabel: ", key)
        right = 0
        wrong = 0
        for v in value:
            if npargmax(v) == key:
                right += 1
            else:
                wrong += 1

        print("\nCorrect predictions: ", right)
        print("Wrong predictions: ", wrong)


def testing_gui(model, results=None):
    # Opens testing GUI
    # In:
    #   results:                dict, label - model output pairs (Not used, just for quick fix)
    #   model:                  model object, model to be used
    
    ModelTester(model)

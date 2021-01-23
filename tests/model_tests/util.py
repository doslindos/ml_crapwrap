from . import npargmax

def parse_results(results):
    y = []
    predictions = []
    labels = len(results.keys())
    for key, result in results.items():
        for logits in result:
            y.append(key)
            if logits.shape[-1] > 1:
                predictions.append(npargmax(logits))
            else:
                predictions.append(logits[0])
            
    labels = [i for i in range(labels)]

    return (predictions, y, labels)

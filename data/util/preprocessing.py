from tensorflow import cast as tfcast, float32 as tffloat32

def normalize_image(data, label=None):
    return tfcast(data, tffloat32) / 255., label

def preprocess_spotify_features(data, label=None):
    #Min Max normalization for the selected dimension
    print(data, label)
    print(dir(data))
    print(data._tf_output)
    print(data.name)
    print(data[0])
    print(dir(data[0]))
    exit()

def preprocess_billboard(data, label=None):
    print(data, label)
    exit()

from .. import tfds_load, tfcast, nparray, tffloat32

def normalize_image(data, label=None):
    return tfcast(data, tffloat32) / 255., label

def normalize_spotify_features(data, label=None):
    #Min Max normalization for the selected dimension
    print(data, label)
    exit()

def normalize_billboard(data, label=None):
    print(data, label)
    exit()

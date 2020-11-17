from .. import tfcast, nparray, tffloat32

def normalize_image(data, label=None):
    return tfcast(data, tffloat32) / 255., label

def normalize_spotify(data, norm_dims=[0, 1, 2, 6, 11]):
    #Min Max normalization for the selected dimension
    data = nparray(data)
    for dim in norm_dims:
        data[:, dim] =  (data[:, dim] - data[:, dim].min()) / (data[:, dim].max() - data[:, dim].min())
    return data

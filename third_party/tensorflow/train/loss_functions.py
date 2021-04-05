from .. import tf

def check_dtypes_match(x, weight):
    
    if x.dtype != weight.dtype:
        x = tf.cast(x, weight.dtype)
    return x

def mean_squared_error(reconstructed_data, original_data):
    original_data = check_dtypes_match(original_data, reconstructed_data)
    return tf.reduce_mean(tf.pow(original_data - reconstructed_data, 2))

def cross_entropy(prediction, true_label):
    true_label = check_dtypes_match(true_label, prediction)
    prediction = tf.clip_by_value(prediction, 1e-9, 1.)
    return tf.reduce_mean(-tf.reduce_sum(true_label * tf.math.log(prediction)))

def cross_entropy_w_sigmoid(prediction, true_label):
    true_label = check_dtypes_match(true_label, prediction)
    prediction = tf.nn.softmax(prediction)
    return tf.reduce_mean(-tf.reduce_sum(true_label * tf.math.log(prediction)))

def keras_sparse_categorical_cross(prediction, true_label):
    true_label = check_dtypes_match(true_label, prediction)
    return tf.keras.losses.sparse_categorical_crossentropy(true_label, prediction, from_logits=True)

def kl_divergence(prediction, target):
    return tf.reduce_mean(tf.reduce_sum(target*tf.math.log(target/(prediction)), axis=1))

from . import tf

def mean_squared_error(reconstructed_data, original_data):
    return tf.reduce_mean(tf.pow(original_data - reconstructed_data, 2))

def cross_entropy(prediction, true_label):
    prediction = tf.clip_by_value(prediction, 1e-9, 1.)
    return tf.reduce_mean(-tf.reduce_sum(true_label * tf.math.log(prediction)))

def cross_entropy_w_sigmoid(prediction, true_label):
    prediction = tf.nn.softmax(prediction)
    return tf.reduce_mean(-tf.reduce_sum(true_label * tf.math.log(prediction)))

def keras_sparse_categorical_cross(prediction, true_label):
    return tf.keras.losses.sparse_categorical_crossentropy(true_label, prediction, from_logits=True)

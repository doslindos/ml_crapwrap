from .. import tf

def are_equal(tensor1, tensor2):
    equal = tf.reduce_all(tf.equal(tensor1, tensor2))
    if equal:
        return True
    else:
        return False

def get_argmax(tensor, dim=1):
    return tf.argmax(tensor, dim).numpy()

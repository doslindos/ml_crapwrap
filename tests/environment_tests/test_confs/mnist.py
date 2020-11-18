# Mnist test command configurations
from .. import Namespace

dataset = Namespace(s='mnist', ff='tfds_fetch', cf=None, name='mnist')
train = Namespace(
                    d='mnist',
                    pf='mnist',
                    m='NeuralNetworks',
                    c='mnist_basic',
                    batch_size=100,
                    epochs=1,
                    learning_rate=0.001,
                    optimization_function=None,
                    loss_function=None,
                    sub_sample=1000
                    )

from third_party.tensorflow.building.util import create_weights, create_bias
from third_party.tensorflow.building.CONV.layer import conv_layer
from third_party.tensorflow.building.DENSE.layer import dense_layer
from third_party.tensorflow.train.training_functions import tf_training_loop
from third_party.tensorflow.train.loss_functions import mean_squared_error, kl_divergence
from third_party.tensorflow.train.optimization import classifier
from plotting.util.plotting import plot_codings

import tensorflow as tf
import numpy as np

class Autoencoder:

    def __init__(self, ws=None):
        # Create autoencoder weights if given
        self.c = {}
        if ws is None:
           self.ws = {
                   'weights': {
                        'conv1': create_weights([3, 3, 1, 32], dtype="float32"),
                        'conv2': create_weights([5, 5, 32, 64], dtype="float32"),
                        'dense': create_weights([14*10*64, 10], dtype="float32")
                        },
                    'bias': {
                        'conv1': create_weights([32], dtype="float32"),
                        'conv2': create_weights([64], dtype="float32"),
                        'dense': create_weights([10], dtype="float32")
                        }
                   } 

           self.trainable_vars = self.get_weights()

        else:
            self.ws = ws

    def get_weights(self):
        def ws(w):
            for value in w.values():
                if isinstance(value, dict):
                    yield from ws(value)
                else:
                    yield value

        return list(ws(self.ws))

    def encoder(self, x, training=False):

        x = conv_layer(
                x, 
                self.ws['weights']['conv1'],
                [1, 1, 1, 1],
                'SAME',
                None,
                self.ws['bias']['conv1'],
                'leaky_relu',
                None,
                0.2,
                training,
                False
                )
        x = conv_layer(
                x, 
                self.ws['weights']['conv2'],
                [1, 2, 3, 1],
                'SAME',
                None,
                self.ws['bias']['conv2'],
                'leaky_relu',
                None,
                False,
                training,
                False
                )
        # FLatten
        x = tf.reshape(x, [-1, 14*10*64])
        x = dense_layer(
                x, 
                self.ws['weights']['dense'],
                self.ws['bias']['dense'],
                None,
                None,
                training,
                False
                )
        
        return x

    def decoder(self, x, training=False):

        x = dense_layer(
                x, 
                tf.transpose(self.ws['weights']['dense']),
                self.ws['bias']['dense'],
                None,
                None,
                training,
                True
                )
        # Reshape
        x = tf.reshape(x, [-1, 14, 10, 64])

        x = conv_layer(
                x, 
                self.ws['weights']['conv2'],
                [1, 2, 3, 1],
                'SAME',
                None,
                self.ws['bias']['conv2'],
                'leaky_relu',
                None,
                False,
                training,
                (x.shape[0], 28,28,32)
                )
        x = conv_layer(
                x, 
                self.ws['weights']['conv1'],
                [1, 1, 1, 1],
                'SAME',
                None,
                self.ws['bias']['conv1'],
                'leaky_relu',
                None,
                0.2,
                training,
                (x.shape[0], 28, 28, 1)
                )

        return x

    def run(self, x, training=False):
        
        x = self.encoder(x, training)
        x = self.decoder(x, training)
        return x

class DeepEmbeddingClustering:

    def __init__(self, num_of_clusters, input_shape, weights=None, alpha=1):
        self.n_clusters = num_of_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.model = Autoencoder(weights)

    def train_Autoencoder(self, datasets):
        tf_training_loop(
            datasets[0].batch(1000),
            datasets[1],
            self.model,
            mean_squared_error,
            classifier,
            tf.optimizers.Adam(0.001),
            1,
            False,
            True,
            False,
            1000
                )


    def cluster(self, data):
        if not hasattr(self, 'centroids'):
            print(self.input_shape)
            self.centroids = tf.Variable(tf.initializers.RandomNormal()([self.n_clusters, self.input_shape]))
        
        #Student t-distribution
        #q_ij interpreted as probability of assigning sample i to cluster j
        #q = 1. / (1. + (tf.reduce_sum(tf.sqrt(tf.expand_dims(data, axis=1) - self.centroids), axis=2) / self.alpha))
        #q **= (self.alpha + 1.) / 2.
        #q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        
        def pairwise_euclidean_dist(a, b):
            p1 = tf.matmul(
                    tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                    tf.ones(shape=(1, self.n_clusters))
                    )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(data.shape[0], 1)),
                transpose_b=True
                ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        dist = pairwise_euclidean_dist(data, self.centroids)
        q = 1.0/(1.0+dist**2/self.alpha)**((self.alpha+1.0)/2.0)
        q = (q/tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def train_centroids(self, train_data, maxiters=8000, update_interval=10, initial_centroids=False):
        

        def encode_data(data):

            for i, d in enumerate(data):
                da = d[0]
                labels = d[1].numpy()
                encoding = self.model.encoder(da).numpy()
                if i == 0:
                    all_encodings = encoding
                    all_labels = labels
                else:
                    all_encodings = np.append(all_encodings, encoding, axis=0)
                    all_labels = np.append(all_labels, labels, axis=0)

            return (all_encodings, all_labels)

        def initialize_centroids():
            from sklearn.cluster import KMeans
            print("Encoding the whole dataset...")
            all_encodings, all_labels = encode_data(train_data.batch(1000))
            print(all_labels.shape)
            print("Starting kmeans...")
            kmeans = KMeans(self.n_clusters, n_init=20)
            kmeans.fit(all_encodings)
            print("Kmeans done...")
            self.centroids =  tf.Variable(kmeans.cluster_centers_)

        if initial_centroids:
            initialize_centroids()

        def target_distribution(q):
            p = q ** 2 / q.sum(0)
            p = p / p.sum(1, keepdims=True)
            return p
        
        def optimize_centroids(x):
            with tf.GradientTape() as g:
                encoding = self.model.encoder(x, True)
                q = self.cluster(encoding)
                p = target_distribution(q.numpy())
                loss = kl_divergence(q, p)
                
                train_vars = self.model.get_weights()
                train_vars += [self.centroids]
                gradients = g.gradient(loss, train_vars)
                optimizer.apply_gradients(zip(gradients, train_vars))

            return loss

        optimizer = tf.optimizers.Adam(0.001)
        train = True
        encodings, labels = encode_data(train_data.batch(1000))
        #initial target dist
        #p = target_distribution(self.cluster(encodings).numpy())
        print("Plotting before training...")
        plot_codings(encodings, labels=labels, show=True)
        while train:
            for epoch in range(5):
                print("Epoch ", epoch)
                for step, batch_x in enumerate(train_data.batch(1000)):
                    if isinstance(batch_x, tuple):
                        x = batch_x[0]
                        if len(batch_x) == 2:
                            y = batch_x[1]
                    elif isinstance(batch_x, dict):
                        x = batch_x['x']
                        if 'y' in batch_x.keys():
                            y = batch_x['y']
                    
                    optimize_centroids(x)
            encodings, labels = encode_data(train_data.batch(1000))
            plot_codings(encodings, labels, show=True)
                
            #update target dists every time the whole dataset is looped through
            #for i, batch_x in enumerate(train_data):
            #    x = batch_x[0]
            #    encoding = model.encoder(x)
            #    q = self.cluster(encoding)
                
            #    if i == 0:
            #        all_qs = q.numpy()
            #    else:
            #        all_qs = np.append(all_qs, q.numpy(), axis=0)

            #    p = traget_distribution(self.cluster(all_qs).numpy())


            if input("Another loop?") == 'n':
                train = False

from models.Manual.manualModels.DeepEmbeddedClustering.MNISTdeepclusteringmodel import DeepEmbeddingClustering

c = {}

def train(datasets):

    # Initialize Cluster model
    DEC = DeepEmbeddingClustering(10, 10)
    #DEC.train_Autoencoder(datasets)
    DEC.train_centroids(datasets[0], initial_centroids=True)

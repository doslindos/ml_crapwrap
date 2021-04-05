from models.Manual.manualModels.DeepEmbeddedClustering.MNISTdeepclusteringmodel import DeepEmbeddingClustering

c = {}

def train(datasets):
    AE = True
    # Initialize Cluster model
    DEC = DeepEmbeddingClustering(10, 10, AE=AE)
    if AE:
        DEC.init_Autoencoder(datasets)
    else:
        DEC.init_Classifier(datasets)
    DEC.train_centroids(datasets[0], initial_centroids=True)

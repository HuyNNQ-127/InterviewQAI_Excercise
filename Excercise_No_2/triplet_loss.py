import numpy as np

# TripletLoss = max(0, sqr(D(anchor, positive))) - sqr(D(anchor, negative)) + e )
# D : Distance metrics(Cosine, Euclid,...)
# e : Minimum distance Hyperparameter

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def triplet_loss(anchor, positive, negative, margin=0.5):
    # cosine similarity
    sim_ap = cosine_similarity(anchor, positive)
    sim_an = cosine_similarity(anchor, negative)

    # euclidean distance
    #sim_ap = euclidean_distance(anchor, positive)**2
    #sim_an = euclidean_distance(anchor, negative)**2
    
    loss = np.maximum(0, sim_ap - sim_an + margin)
    return loss

# demo
anchor = np.array([1.0, 0.5, -0.2])
positive = np.array([1.1, 0.6, -0.1])
negative = np.array([2, 1.6, 1])

loss = triplet_loss(anchor, positive, negative)
print(f'Triplet Loss: {loss}')

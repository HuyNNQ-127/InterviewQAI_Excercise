import numpy as np

#Calculate every single triplet loss pair for every single positive, after that average the results

def cosine_similarity(a, b):

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def triplet_loss(anchor, positives, negatives, margin=0.2):
    loss = 0
    num_positives = len(positives)
    num_negatives = len(negatives)
    
    for pos in positives:
        sim_ap = cosine_similarity(anchor, pos)
        for neg in negatives:
            sim_an = cosine_similarity(anchor, neg)
            loss += np.maximum(0, sim_ap - sim_an + margin)
    
    loss = loss / (num_positives * num_negatives)
    return loss

# demo
anchor = np.array([1.0, 0.5, -0.2])

positives = [
    np.array([1.1, 0.6, -0.1]), 
    np.array([0.9, 0.4, -0.3])]

negatives = [
    np.array([-0.5, -0.3, 0.8]),
    np.array([-0.4, 0.2, -0.6]),
    np.array([0.3, -0.7, 0.5]),
    np.array([0.1, 0.1, -0.8]),
    np.array([-0.2, -0.4, 0.6])
]

# Compute triplet loss
loss = triplet_loss(anchor, positives, negatives)
print(f'Triplet Loss: {loss}')
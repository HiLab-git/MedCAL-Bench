from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def read_tsv(file_path):
    scores = []
    with open(file_path, 'r') as file:
        for line in file:
            scores.append(float(line.strip()))
    return np.array(scores)

def compute_typicality(cluster_feats):
     distances = cdist(cluster_feats, cluster_feats, metric='euclidean')
     avg_distances = np.mean(distances, axis=1)
     typicalities = 1 / avg_distances
     return typicalities
 
def estimate_delta(embedding, num_classes, alpha=0.95):
    kmeans = KMeans(n_clusters=num_classes)
    cluster_labels = kmeans.fit_predict(embedding)
    
    dist_matrix = distance_matrix(embedding, embedding)
    
    delta_values = np.linspace(0, np.max(dist_matrix), num=100)
    best_delta = 0
    for delta in delta_values:
        pure_balls = 0
        total_balls = 0
        
        for i in range(len(embedding)):
            neighbors = np.where(dist_matrix[i] <= delta)[0]
            if len(neighbors) > 0:
                if np.all(cluster_labels[neighbors] == cluster_labels[i]):
                    pure_balls += 1
                total_balls += 1
        
        purity = pure_balls / total_balls
        if purity >= alpha:
            best_delta = delta
        else:
            break
    
    return best_delta

def compute_information_density(cluster_feats):
    similarity_matrix = cosine_similarity(cluster_feats)
    densities = similarity_matrix.mean(axis=1)
    return densities
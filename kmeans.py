# from sklearn.cluster import KMeans
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from fast_pytorch_kmeans import KMeans
import torch
from queue import PriorityQueue

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything()

class EquallySizedKMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def fit(self, data):
        # Compute desired cluster size
        desired_size = len(data) // self.n_clusters
        soft_upper_limit_min = int(desired_size)
        soft_upper_limit_max = int(desired_size) + 1
      
        # Initialize means with k-means++
        init_kmeans = KMeans(n_clusters=self.n_clusters)
        labels = init_kmeans.fit_predict(data)
        centroids = init_kmeans.centroids
        labels = labels.cpu()
        centroids = centroids.cpu()
        data = data.cpu()
        for _ in range(10):  # Max 10 iterations, can be changed
            # Compute current cluster means
            tic = time.time() 
            centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(self.n_clusters)],dim=0)
            toc = time.time()
            # print("indexing time: ", toc-tic)
            distances = ((data - centroids[:, np.newaxis])**2).sum(dim=2)
            
            # Compute the delta for sorting
            min_distances = distances.min(dim=0)[0]
          
            
            # second_min_distances = np.partition(distances, 1, axis=0)[1]
            second_min_distances = distances.max(dim=0)[0]
         
            deltas = second_min_distances - min_distances

            # Sort elements by delta
            sorted_idx = torch.argsort(deltas,descending=True)#[::-1] #if delta is large, then they should be sort first
            
            # Reassign labels
            labels = torch.empty_like(labels)
            # print(labels)
            cluster_sizes = torch.zeros(self.n_clusters, dtype=int)
            for idx in sorted_idx: # 1000
                best_clusters = torch.argsort(distances[:, idx],descending=False) # distance from close to far for point idx
                all_full = True
                for cluster in best_clusters: # 10
                    if cluster_sizes[cluster] < soft_upper_limit_min:
                        all_full = False
                        labels[idx] = cluster
                        cluster_sizes[cluster] += 1
                        break
                if all_full:
                    for cluster in best_clusters:
                        if cluster_sizes[cluster] < soft_upper_limit_max:
                            labels[idx] = cluster
                            cluster_sizes[cluster] += 1
                            break
        centroids = centroids.cpu().numpy()
        labels = labels.cpu().numpy()
        return centroids, labels
    
    
    
class EquallySizedKMeans_CPU:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, data):
        # Compute desired cluster size
        desired_size = len(data) // self.n_clusters
        soft_upper_limit_min = int(desired_size)
        soft_upper_limit_max = int(desired_size) + 1
        
        # Initialize means with k-means++
        init_kmeans = KMeans(n_clusters=self.n_clusters)
        labels = init_kmeans.fit_predict(data)
        centroids = init_kmeans.centroids
        
        # Assign points based on the initialization method described
        # labels = np.argmin(((data - centroids[:, np.newaxis])**2).sum(axis=2), axis=0)
        data = data.cpu().numpy()
        labels = labels.cpu().numpy()
        centroids = centroids.cpu().numpy()
        for _ in range(10):  # Max 10 iterations, can be changed
            # Compute current cluster means
            # tic = time.time()
            centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)]) #(n_clusters, 2)
            # toc = time.time()
            # print("indexing time: ", toc-tic)
            # Compute distances to cluster means
            distances = ((data - centroids[:, np.newaxis])**2).sum(axis=2)
            
            # Compute the delta for sorting
            min_distances = distances.min(axis=0)
            # print(min_distances.shape) # 1000
            
            # second_min_distances = np.partition(distances, 1, axis=0)[1]
            second_min_distances = distances.max(axis=0)
            deltas = second_min_distances - min_distances

            # Sort elements by delta
            sorted_idx = np.argsort(deltas)[::-1] #if delta is large, then they should be sort first
            
            # Reassign labels
            labels = np.empty_like(labels)
            # print(labels)
            cluster_sizes = np.zeros(self.n_clusters, dtype=int)
            for idx in sorted_idx: # 1000
                best_clusters = np.argsort(distances[:, idx]) # distance from close to far for point idx
                all_full = True
                for cluster in best_clusters: # 10
                    if cluster_sizes[cluster] < soft_upper_limit_min:
                        all_full = False
                        labels[idx] = cluster
                        cluster_sizes[cluster] += 1
                        break
                if all_full:
                    for cluster in best_clusters:
                        if cluster_sizes[cluster] < soft_upper_limit_max:
                            labels[idx] = cluster
                            cluster_sizes[cluster] += 1
                            break
        return centroids, labels

X = torch.rand(1000, 2).to(device)
n_cluster = 11

tic = time.time()
for _ in range(10):
    k = EquallySizedKMeans_CPU(n_cluster)
    c,l = k.fit(X)
toc = time.time()
print(toc-tic)

tic = time.time()
for _ in range(10):
    k = EquallySizedKMeans(n_cluster)
    c,l = k.fit(X)
toc = time.time()
print(toc-tic)



# print(l)
# plt.figure()
# X = X.cpu().numpy()
# c = np.random.rand(n_cluster,3)
# for i in range(n_cluster):
#     plt.scatter(X[l==i,0], X[l==i,1],c=c[i])
# plt.show()
# # print(l.shape)
# plt.figure()
# plt.hist(l, bins=n_cluster)
# plt.show()



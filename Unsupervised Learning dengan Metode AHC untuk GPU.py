# Metode Unsupervised Machine Leraning menggunakan Agglomerative Hierarhical Clustering (AHC)
# Nama : Laksamana Sulthan Alam .S
# NIM : 19650098
# Kelas : Kecerdasan Buatan - E

# Importing the libraries
import numpy as np
import matplotlib.pyplot as lihat
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Dataset diambil dari data yang sudah dibersihkan
dataset = pd.read_csv('D:\Semester 5\Kecerdasan Buatan E\PROJEK AKHIR\Program\data_processing.csv')
X = dataset.iloc[:, [3, 4]].values

# Tampilan Dendogram dari dataset GPU
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
lihat.title('Dendrogram')
lihat.xlabel('Produk GPU')
lihat.ylabel('Euclidean distances')
lihat.show()

# Proses Training Metode Agglomerative Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Kluster divisualisasikan menjadi lima jenis
lihat.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
lihat.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
lihat.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
lihat.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
lihat.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

#Legenda/Keterangan didalam Clusternya
lihat.title('Clusters Harga GPU')
lihat.xlabel('Harga GPU ($)')
lihat.ylabel('Kualitas GPU')

#show dan legenda
lihat.legend()
lihat.show()
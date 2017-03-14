import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

#Extracting the Latitude and Longitude features
fields = ['X','Y']
df = pd.read_csv('test.csv', usecols = fields)
numpyMatrix = df.as_matrix()

#Scaling Features
for i in range(len(numpyMatrix)):
	x = numpyMatrix[i][0]
	x *= (-1)
	x %= 122
	x *= 100
	numpyMatrix[i][0] = x

	y = numpyMatrix[i][1]
	y %= 37
	y *= 100
	numpyMatrix[i][1] = y


#setting limits of the graph
plt.xlim(37.5, 52.5)
plt.ylim(70, 82.5)

#setting labels
plt.xlabel('Latitude')
plt.ylabel('Longitude')


#Training the model
kmeans = KMeans(init = 'k-means++', n_clusters = 10, random_state = 56)
kmeans.fit(numpyMatrix)


#printing cluster centers
print('Printing Cluster Centers:')
print(kmeans.cluster_centers_)

#plotting clusters and centroids
colors = [i for i in kmeans.labels_]
plt.scatter(numpyMatrix[:,0], numpyMatrix[:,1], s = 5, c = colors)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='black', zorder=10)
plt.title('K-Means Clustering')
plt.figure(1)
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import AgglomerativeClustering

#Extracting the Latitude and Longitude features
fields = ['X','Y']
df = pd.read_csv('test.csv', usecols = fields)
numpyMatrix = df.as_matrix()

#Random Sampling is done on the data 
randomsample = []
for i in range(0, len(numpyMatrix)):
	a = random.uniform(0,50)
	if(a <= 1):
		randomsample.append(numpyMatrix[i])

randomsample = np.asarray(randomsample)

numpyMatrix = randomsample

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
model = AgglomerativeClustering(n_clusters=10)
model.fit(numpyMatrix)

#plotting clusters and centroids
colors = [i for i in model.labels_]
plt.scatter(numpyMatrix[:,0], numpyMatrix[:,1],marker = 'o' ,s = 20, c = colors)
plt.title('Hierarchial Clustering')
plt.figure(1)
plt.show()



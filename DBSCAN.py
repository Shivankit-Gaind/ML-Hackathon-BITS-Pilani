import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import DBSCAN


#Extracting the Latitude and Longitude features
fields = ['X','Y']
df = pd.read_csv('test.csv', usecols = fields)
numpyMatrix = df.as_matrix()

#no. of points in original data
print(len(numpyMatrix))

#Random Sampling is done on the data 
randomsample = []
for i in range(0, len(numpyMatrix)):
	a = random.uniform(0, 5)
	if(a <= 1):
		randomsample.append(numpyMatrix[i])

randomsample = np.asarray(randomsample)

#no. of points in random sample
print(len(randomsample))


#Scaling Features
for i in range(len(randomsample)):
	x = numpyMatrix[i][0]
	x *= (-1)
	x %= 122
	x *= 100
	randomsample[i][0] = x

	y = randomsample[i][1]
	y %= 37
	y *= 100
	randomsample[i][1] = y

#setting limits of the graph
plt.xlim(37.5, 52.5)
plt.ylim(70, 82.5)

#setting labels
plt.xlabel('Latitude')
plt.ylabel('Longitude')

#Training the model
dbscan = DBSCAN(eps = 0.03, min_samples = 50)
dbscan.fit(randomsample)

print(dbscan.labels_)

#plotting clusters
colors = [i for i in dbscan.labels_]
plt.scatter(randomsample[:,0], randomsample[:,1], s = 0.01, c = colors)
plt.title('Density Based Clustering')
plt.figure(1)
plt.show()


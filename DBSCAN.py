import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
from matplotlib import cm
from sklearn.cluster import DBSCAN
style.use('ggplot')
import pylab

fields = ['DayOfWeek', 'X', 'Y']

df = pd.read_csv('test.csv', usecols = fields)


print(df.head())
print(len(df))

x = []
y = []

length = len(df)
for i in range(0, length):
	a = random.uniform(0, 5)
	if(a <= 1):
		x.append(df['X'][i])
		y.append(df['Y'][i])

x = np.asarray(x)#[:10000]
y = np.asarray(y)#[:10000]

print(str(len(x)) + '   ' + str(len(y)))

# read in numpy

x1 = []
for i in x:
    i = i * (-1)
    i = i % 122
    i *= 100
    x1.append(i)

y1 = []
for i in y:
    i = i % 37
    i *= 100
    y1.append(i)

x1 = np.asarray(x1)
y1 = np.asarray(y1)

# sk learn trandform
combined = np.vstack((x1, y1)).T

plt.xlim(35, 55)
plt.ylim(65, 85)

dbscan = DBSCAN(eps = 0.05, min_samples = 3)
dbscan.fit(combined)

colors = [int(i % 23) for i in dbscan.labels_]
print(len(set(colors)))
plt.scatter(x1, y1, s = 0.01, c = colors)

plt.figure(1)
plt.show()

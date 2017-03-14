import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
from matplotlib import cm
from sklearn.cluster import KMeans
style.use('ggplot')

fields = ['DayOfWeek', 'X', 'Y']

df = pd.read_csv('test.csv', usecols = fields)

print(df.head())
print(len(df))
#print(df['X'][0])

#fig = plt.figure()
#ax = fig.gca(projection = '3d')

x = []
y = []
#z = []

for row in df['X']:
    x.append(row)
for row in df['Y']:
    y.append(row)
'''for row in df['DayOfWeek']:
    if(row == 'Sunday'):
        z.append(7)
    elif(row == 'Monday'):
        z.append(1)
    elif(row == 'Tuesday'):
        z.append(2)
    elif(row == 'Wednesday'):
        z.append(3)
    elif(row == 'Thursday'):
        z.append(4)
    elif(row == 'Friday'):
        z.append(5)
    elif(row == 'Saturday'):
        z.append(6)'''

x = np.asarray(x)
y = np.asarray(y)
#z = np.asarray(z)

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

combined = np.vstack((x1, y1)).T

#print(len(x1))
#print(len(y1))

plt.xlim(35, 55)
plt.ylim(65, 85)

#trialplot = ax.scatter(x1, y1, s = area)
plt.scatter(x1, y1, s = 0.01)

#df.plot()
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('DayOfWeek')
#plt.show()

kmeans = KMeans(n_clusters = 10, random_state = 56)
kmeans.fit(combined)
print(kmeans.cluster_centers_)

plt.figure(1)
plt.show(kmeans)

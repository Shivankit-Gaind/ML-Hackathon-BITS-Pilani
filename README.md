# ML-Hackathon-BITS-Pilani
This project involves the use of unsupervised analysis on San Francisco Crime Rate Dataset, Clustering, for conducting an introductory hands-on session on Machine Learning in BITS Pilani, The Machine Learning Experience, a week before the Technical Fest APOGEE, 2017. The crime locations are clustered on the basis of their latitudes and longitudes. Three different clustering techniques have been used to produce three different results, methods namely:

  - K-Means Clustering
  - Agglomerative Clustering (Bottom Up approach to the Hierarchichal Clustering Technique)
  - DBSCAN

The results of each of the clustering algorithms are given as images (which are plots generated using matplotlib - a python library).

- Python Libraries used:
    - Scikit-learn
    - Pandas
    - Numpy
    - Matplotlib
    - Pyplot
    - iPython Notebooks

- Pre-processing techniques used
  - Random Sampling for computationally expensive algorithms, i.e. Agglomerative Clustering and DBSCAN.
  - Scaling of data to appropriate limits for better visualization.
  
- Type of data :
  - Lattitude (Continuous, Ratio)
  - Longitude (Continuous, Ratio)

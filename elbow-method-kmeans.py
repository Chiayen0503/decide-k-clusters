#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:52:00 2018

@author: ChiaYen
"""

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


dataRaw = [];
DataFile = open("MopsiLocations2012-Joensuu.txt", "r")

while True:
    theline = DataFile.readline()
    if len(theline)== 0:
        break
    readData = theline.split(" ") #use " " space to seperate two columes 
    
    for pos in range(len(readData)):
        readData[pos] = float(readData[pos]);
    dataRaw.append(readData)

DataFile.close()


data = np.array(dataRaw)

X = data
distorsions = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
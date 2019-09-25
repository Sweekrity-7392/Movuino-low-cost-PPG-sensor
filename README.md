# Project-1

HI, I am Sweekrity from CRI, Paris. I am working on low cost wearables and trying to obtain good signals with well defined peaks and comparable to ECG. 

Denoising of signals from movuino
#%%
import numpy as np
import matplotlib.pylab as plt
import peakutils as pk
from peakutils.plot import plot as pplot

f = open('movuino-recording-2.csv','r')
F = f.read()
#print(F)
G = F.split('\n')
G = G[1:]
A = [element.split(',') for element in G]
A = [[float(x) for x in row] for row in A]
#print(A)
#print(np.shape(A))
B = np.transpose(A)
#print(B)
plt.plot(B[0],B[3])
plt.show()

#finding peaks using peakutils
peak_indexes = pk.indexes(B[3])
pplot(B[1], peak_indexes)


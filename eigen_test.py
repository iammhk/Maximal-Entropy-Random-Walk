import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
size=33
pi = np.pi
A = np.array([[ 0, .3, .2],
              [.7,  0, .1],
              [.8, .9,  0]])
A = np.zeros([size,size])

for i in range(size):
    for j in range(size):
        A[i,j]= np.sin(pi * i / size) * np.sin(pi * j / size)

plt.imshow(A)
plt.show()

eig_values, eig_vectors = la.eig(A)

eig_values = np.real_if_close(eig_values, tol=1)
eig_vectors = np.real_if_close(eig_vectors, tol=1)


max_Evalue = np.amax(eig_values)
max_index = np.where(eig_values == np.amax(eig_values))
max_Evector = eig_vectors.T[0]

print(max_Evalue)
print(max_Evector)
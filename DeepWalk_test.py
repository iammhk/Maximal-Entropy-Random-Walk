import numpy as np
import random
import scipy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(10)
def adj(s):
    vector = np.zeros(s)
    vector[1]= 1
    vector[-1]= 1
    offdi = la.circulant(vector) #for periodic boundary
    #offdi = la.toeplitz([0,1,0,0,0]) #for non-periodic boundary
    I = np.eye(s)
    a = np.kron(offdi, I) + np.kron(I, offdi)
    #print(a)
    return a

def NN_prob(node):
    nbr = list(G.neighbors(node))
    #print("Total neighbours:", len(nbr))
    prob = np.zeros((len(nbr)))
    for j in range(len(nbr)):
            prob[j] = abs((max_Evector[nbr[j]])/( max_Evalue * max_Evector[node]))
    #print("Sum of prob:",sum(prob))
    prob[:]= prob[:]/ sum(prob)
    return prob

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def get_rw(node, path_length):
    #print(node)
    random_walk = [node]
    values[node] += 1
    #print(list(G.neighbors(node)))
    for i in range(path_length-1):
        temp = list(G.neighbors(node))
        #temp = list(set(temp) - set(random_walk))
        print(len(random_walk),len(temp))
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        values[random_node]+=1
        node = random_node

    return random_walk

def get_merw(node, path_length):
    random_walk = [node]
    values[node] += 1
    for i in range(path_length-1):
        temp = list(G.neighbors(node))
        #print("Node:", node)
        #print("NBRs:",temp)
        #temp = list(set(temp) - set(random_walk))
        #print(len(random_walk),len(temp))
        if len(temp) == 0:
            break

        random_node = np.random.choice(temp, p=NN_prob(node))
        random_walk.append(random_node)
        values[random_node]+=1
        node = random_node


    return random_walk

#Constructing a Graph
#G = nx.karate_club_graph()
dimension = 16
G = nx.grid_2d_graph(dimension, dimension)
print("Total nodes: ",len(G))

values = [0 for node in G.nodes()]
size =len(G)
A = nx.to_numpy_matrix(G)
print("Symmetric:", check_symmetric(A))
eig_values, eig_vectors = la.eig(A)

eig_values = np.real_if_close(eig_values, tol=1)
eig_vectors = np.real_if_close(eig_vectors, tol=1)

max_Evalue = np.amax(eig_values)
# max_index = np.where(eig_values == np.amax(eig_values))
max_Evector = eig_vectors.T[0]

pie_star = np.zeros((size))
pie_starD = np.zeros((size))
pie_sum = 0


for i in range(size):
    pie_starD[i] = max_Evalue * max_Evalue
    pie_star[i] = max_Evector[i] * max_Evector[i]
    pie_sum += pie_star[i]

print("Pie sum:", pie_sum)

for i in range(1):
    print("Walk:", get_rw(np.random.randint(0,dimension), 100))


#print(values)
pos = nx.spring_layout(G)
nx.draw(G, pos,  cmap=plt.get_cmap('Blues'), node_color = values, with_labels = True)
#plt.colorbar(values)
plt.show()
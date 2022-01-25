import numpy as np
import random
import scipy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(10)
dimension = 16
defects = 14

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

def label_lin(pos):
    #print(pos)
    return (pos[0] * dimension) + pos[1]

#print("2d to 1d (1,2):", label_lin([1,2]))


def label_2d(pos):
    #print(pos)
    return int(pos/dimension), int(pos%dimension)

#print("1d to 2d (18):", label_2d(18))

def NN_prob(node):
    nbr = list(G.neighbors(node))
    #print("Total neighbours:", len(nbr))
    prob = np.zeros((len(nbr)))
    x0, y0 = node[0], node[1]
    for j in range(len(nbr)):
        x1,y1 = nbr[j][0],nbr[j][1]
        prob[j] = abs((max_Evector[label_lin(nbr[j])])/( max_Evalue * max_Evector[label_lin(node)]))
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
        #print(len(random_walk),len(temp))
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
        nbr= np.zeros(len(temp))
        for k in range(len(temp)):
            nbr[k] = label_lin(temp[k])
        #print("Node:", node)
        #print("NBRs:",temp)
        #temp = list(set(temp) - set(random_walk))
        #print(len(random_walk),len(temp))
        if len(temp) == 0:
            break

        random_node = np.random.choice(nbr, p=NN_prob(node))
        pos = label_2d(random_node)
        #print(pos)
        random_walk.append(pos)
        values[pos]+=1
        node = pos

    return random_walk

#Constructing a Graph

G = nx.grid_graph(dim=(dimension, dimension), periodic=False)

#G.remove_edge((2, 5), (3, 5))

labels = dict( ((i, j), i * dimension + j) for i, j in G.nodes() )

coordx,coordy=[],[]

for m in range (defects):
    #x,y = np.random.randint(1, dimension-1),np.random.randint(1, dimension-1)
    #print(x,y)
    #if x not in coordx and y not in coordy:
        x,y= [int(x) for x in input("Enter two value: ").split()]
        try:
            G.remove_edge((x, y), (x + 1, y))
            G.remove_edge((x, y), (x - 1, y))
            G.remove_edge((x, y), (x, y + 1))
            G.remove_edge((x, y), (x, y - 1))
        except: pass

    #coordx.append([x])
    #coordy.append([y])

print("Total nodes: ",len(G))

#values = [0 for node in G.nodes()]
values= np.zeros((dimension,dimension))
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

#print("Pie sum:", pie_sum)

for i in range(1):
    print("Walk:", get_rw((np.random.randint(0,dimension),np.random.randint(0,dimension)), 10000))


#print(values)
pos = dict( (n, n) for n in G.nodes() )
nx.draw(G, pos,  cmap=plt.get_cmap('Blues'), node_color = values, labels=labels)
#plt.colorbar(values)
plt.show()
import networkx as nx
import pandas as pd
import numpy as np
import random
import scipy.linalg as la
from tqdm import tqdm
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from pylab import figure
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

np.random.seed(14)
df = pd.read_csv("phy.tsv", sep = "\t")
df.head()

G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.Graph())
crossings = list(G.nodes(data=True))
num_dict = {i: val[0] for i, val in enumerate(crossings)}
num_dict.update({ val[0]:i for i, val in enumerate(crossings)})

#print(num_dict[5])

def NN_prob(node):
    nbr = list(G.neighbors(num_dict[node]))
    #print("Total neighbours:", nbr)
    prob = np.zeros((len(nbr)))
    for j in range(len(nbr)):
            prob[j] = abs((max_Evector[num_dict[nbr[j]]])/( max_Evalue * max_Evector[node]))
    #print("Sum of prob:",sum(prob))
    prob[:]= prob[:]/ sum(prob)
    return prob

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_rw(node, path_length):
    random_walk = [node]

    for i in range(path_length - 1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node

    return random_walk

def get_merw(node, path_length):
    random_walk = [num_dict[node]]
    values[node] += 1
    for i in range(path_length-1):
        temp = list(G.neighbors(num_dict[node]))
        #print("Node:", node)
        #print("NBRs:",temp)
        #temp = list(set(temp) - set(random_walk))
        #print(len(random_walk),len(temp))
        if len(temp) == 0:
            break

        random_node = np.random.choice(temp, p=NN_prob(node))
        #print(random_node)
        random_walk.append(random_node)
        values[num_dict[random_node]] += 1
        node = num_dict[random_node]


    return random_walk

def plot_nodes(word_list):
    X = model.wv[word_list]

    # reduce dimensions to 2
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    plt.figure(figsize=(12, 9))
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.show()


def plot_nodes3d(word_list):
    X = model.wv[word_list]

    # reduce dimensions to 2
    pca = PCA(n_components=3)
    result = pca.fit_transform(X)

    fig = figure()
    ax = fig.add_subplot(projection='3d')

    # create a scatter plot of the projection
    #plt.scatter(result[:, 0], result[:, 1], result[:, 2])
    for i, word in enumerate(word_list):
        ax.scatter(result[i, 0], result[i, 1], result[i, 2], color='b')
        ax.text(result[i, 0], result[i, 1], result[i, 2], '%s' % (word), size=8, zorder=1,
                color='k')


    plt.show()

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

# get list of all nodes from the graph

#H=nx.convert_node_labels_to_integers(G)
all_nodes = list(G.nodes())
random_walks = []
N=0
for n in tqdm(all_nodes):
    #print("Node:", n)
    for i in range(5):
        random_walks.append(get_merw(N, 10))
    N+=1
# count of sequences
print(len(random_walks))

# train skip-gram (word2vec) model
model = Word2Vec(window = 4, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14, compute_loss=True, size=128)

model.build_vocab(random_walks, progress_per=2)

model.train(random_walks, total_examples = model.corpus_count, epochs=10, report_delay=1)
training_loss = model.get_latest_training_loss()
print("Loss:",training_loss)

#print(model.wv.similar_by_word('spacetime'))

terms=['spacetime', 'special relativity', 'minkowski space', 'poincaré group', 'history of electromagnetism', 'electromagnetic spectrum', 'permittivity', 'solenoid', 	'big bang', 'history of astronomy',	'cosmology', 'principia mathematica', 'golden ratio', 'introduction to mathematical philosophy']

#terms=[]
#for i in range(20):terms.append(all_nodes[np.random.randint(len(all_nodes))])

plot_nodes(terms)
# import libraries
import networkx as nx
from karateclub import DeepWalk
import matplotlib.pyplot as plt
# load the karate club dataset
G = nx.karate_club_graph()
#G = nx.complete_graph(5)
nx.draw(G)
A = nx.to_numpy_matrix(G)
print(A)
# load the DeepWalk model and set parameters
dw = DeepWalk(dimensions=4, walk_length=4, walk_number=10, window_size=4, epochs=1)
# fit the model
dw.fit(G)
# extract embeddings
embedding = dw.get_embedding()
print (embedding)
plt.show()
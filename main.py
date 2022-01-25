import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from sympy import Matrix
size=16 #please keep an odd number
q=0.1
n = 5000 #n is the number of steps(increase in the value of n increses the compelxity of graph)

midpnt = 1 * (size - 1) / 2
projection=[]
walk_x = np.full ((n), midpnt ) # x and y are arrays which store the coordinates of the position
walk_y = np.full ((n), midpnt )

direction=["U","D","L","R"] # Assuming the four directions of movement.
#np.random.seed(10)

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


def make_matrix(rows, cols):
    n = rows*cols
    M = np.zeros([n,n])
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: M[i-1,i] = M[i,i-1] = 1
            # Two outer diagonals
            if r > 0: M[i-cols,i] = M[i,i-cols] = 1
    return (M)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def add_defects(a,d):
    D = int(d * size * size)

    print(" Total Defects: ",D )
    for i in range(D):
        x = np.random.randint(0,size*size)
        #print(x)
        a[x, :] = 0
        a[:, x] = 0
    return a

def adj2graph(a):
    zero_row = np.where(~a.any(axis=0))[0]
    zero_col = np.where(~a.any(axis=1))[0]
    lat = np.ones([size,size])
    #print(zero_row, zero_col)
    for i in range(zero_row.shape[0]):
        row = int(zero_row[i] / size)
        col = int(zero_col[i] % size)
        #print("Defect at: ", row, col)
        lat[row,col]=0
    return lat

def NN_prob(x0,y0):
    i = int(x0 + (size*(y0)))
    #print("x,y,i:", x0, y0,i)

    ind = np.asarray(np.nonzero(A[i]))[0]
    #print("Total neighbours:", len(ind))
    prob = [0,0,0,0]
    #print("Non-Zero Indices for :", x0, y0)
    for j in range(len(ind)):
        nbr = ind[j]
        val = A[i, nbr]
        x1 = nbr % size
        y1 = int(nbr / size)
        dx = x1 - x0
        dy = y1 - y0
        move = ""
        if dx == -1:
            move = "L"
            prob[2] = abs((val* max_Evector[nbr])/( max_Evalue * max_Evector[i]))
        elif dx == 1:
            move = "R"
            prob[3] = abs((val* max_Evector[nbr])/( max_Evalue * max_Evector[i]))
        elif dy == 1:
            move = "D"
            prob[1] = abs((val* max_Evector[nbr])/( max_Evalue * max_Evector[i]))
        elif dy == -1:
            move = "U"
            prob[0] = abs((val* max_Evector[nbr])/( max_Evalue * max_Evector[i]))
        elif dx > 1:
            move = "L"
            prob[2] = abs((val* max_Evector[nbr])/( max_Evalue * max_Evector[i]))
        elif dx < -1:
            move = "R"
            prob[3] = abs((val* max_Evector[nbr])/( max_Evalue * max_Evector[i]))
        elif dy > 1:
            move = "U"
            prob[0] = abs((val* max_Evector[nbr])/( max_Evalue * max_Evector[i]))
        elif dy < -1:
            move = "D"
            prob[1] = abs((val* max_Evector[nbr])/( max_Evalue * max_Evector[i]))
        else:
            move = "Other"
        #print(x1, y1)
        #print("dx,dy:", dx, dy, move)
    #print(prob[0]+prob[1]+prob[2]+prob[3])
    prob[:]= prob[:]/ (prob[0]+prob[1]+prob[2]+prob[3])
    return prob


A = adj(size)
#A_star =
#print(A)
A = add_defects(A, q)
#print(defect_A)
Lattice = adj2graph(A)

print("Symmetric:", check_symmetric(A))

diag_adj = np.diag(A)

eig_values, eig_vectors = la.eig(A)

eig_values = np.real_if_close(eig_values, tol=1)
eig_vectors = np.real_if_close(eig_vectors, tol=1)

max_Evalue = np.amax(eig_values)
max_index = np.where(eig_values == np.amax(eig_values))
max_Evector = eig_vectors.T[0]

pie_star= np.zeros((size*size))
pie_starD= np.zeros((size*size))
pie_sum=0

for i in range (size*size):
    pie_starD[i] = max_Evalue * max_Evalue
    pie_star[i]= max_Evector[i] * max_Evector[i]
    pie_sum += pie_star[i]
pie_star=np.reshape(pie_star,[size,size])
pie_starD=np.reshape(pie_starD,[size,size])

print("Pie sum:", pie_sum)
fig, axs = plt.subplots(1, 2)
axs[0].imshow(Lattice.T)
axs[0].set_title("Lattice")
axs[1].imshow(pie_star.T)
axs[1].set_title("Pi*")
fig.tight_layout()

print(Lattice)
print(pie_star)

#projection.append(1)
for i in range(1, n):
    move_prob = NN_prob(int(walk_x[i-1]%size),int(walk_y[i-1]%size))
    '''connections=np.count_nonzero(move_prob)
    for j in range(4):
        if(move_prob[j] != 0): move_prob[j]=1/connections'''
    print(move_prob)
    step = np.random.choice(direction, p=move_prob) #Randomly choosing the direction for MERW.
    #step = np.random.choice(direction)  # Randomly choosing the direction for RW.
    #print("MOVED:",step)
    if step == "R": #updating the direction with respect to the direction of motion choosen.
        walk_x[i] = walk_x[i - 1] + 1
        walk_y[i] = walk_y[i - 1]
    elif step == "L":
        walk_x[i] = walk_x[i - 1] - 1
        walk_y[i] = walk_y[i - 1]
    elif step == "U":
        walk_x[i] = walk_x[i - 1]
        walk_y[i] = walk_y[i - 1] - 1
    else:
        walk_x[i] = walk_x[i - 1]
        walk_y[i] = walk_y[i - 1] + 1

    projection.append(walk_y[i]/max(walk_x[i],0.1))
    #projection.append(projection[-1]+walk_y[i]-walk_x[i])

fig, ax = plt.subplots(2,1, figsize=[10, 5])
fig.tight_layout()
lattice0= np.zeros((size,size))
lattice= np.zeros((size,size))
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
lattices = []
oneD_walk = []
for t in range(n):
    step_x = int(walk_x[t] % size)
    step_y = int(walk_y[t] % size)
    #print(t)
    lattice[step_x, step_y] += 0.01
    lat = ax[0].imshow(lattice, animated=True)
    ax[1].clear()
    if t == 0:
        ax[0].imshow(lattice)  # show an initial one first
    lattices.append([lat])

ani = animation.ArtistAnimation(fig,lattices, interval=25, blit=True, repeat_delay=1000)
        #print(val)'''

'''xs = []
ys = []
def animate(i):
    step_x = int(walk_x[i] % size)
    step_y = int(walk_y[i] % size)
    lattice[step_x, step_y] += 0.01
    xs.append(i)
    ys.append(projection[i])

    #ax.clear()
    ax[0].imshow(lattice, animated=True)
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)
    ax[1].plot(xs, ys)
    if i == 0:
        ax[0].imshow(lattice)  # show an initial one first
    plt.xlabel('Time')
    plt.ylabel('Price')


ani = animation.FuncAnimation(fig, animate, interval=200)
'''
plt.show()
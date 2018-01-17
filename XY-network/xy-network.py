import numpy as np
import scipy as sp
from math import pi
from random import *
from numpy import cos, sin
import matplotlib.pyplot as plt
import networkx as nx



MC_step = 500
thermal_steps = 2000


G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4, 5])
#G.add_edge(0,1)
G.add_edges_from([(0, 1, {'J' : -10}),
                 (1, 2, {'J' : -10}),
                 (2, 3, {'J' : -10}),
                 (3, 4, {'J' : -10}),
                 (3, 5, {'J' : -10}),
                 (2, 6, {'J' : -10}),
                 (4, 7, {'J' : -10}) 
                 ])
                 
#H.add_edges_from([(0,1, {'J' : 10})])
 #A=np.matrix([[0,1,1],[1,0,0],[1,0,0]])
#G=nx.from_numpy_matrix(A)

n_spins = len(G.nodes())
theta = np.zeros(n_spins)   


def initialize_rand():
    for i in range(n_spins):
        theta[i] = (2*pi)*random()
             

def energy_in_site(node):
    neighbours = G.adjacency_list()[node]
    return sum([ -G[node][i]['J']*cos(theta[node] - theta[i]) for i in neighbours])
    
    
def energy_total():
    E = 0
    for i in range(n_spins):
            E += energy_in_site(i) 
    return E/2
    
def flip():
    beta = 20
    node = np.random.randint(n_spins)
    
    e_save = energy_in_site(node)
    theta_save = theta[node]
    
    theta[node] += (2*pi)*random()
    e = energy_in_site(node)
    
    delta_e = e - e_save
    if delta_e < 0:
        return True
    else:
        flip_ratio = np.exp(-beta*delta_e)
        if flip_ratio > random():
            return True
        else:
            theta[node] = theta_save
            return False
             
def oneMCstep():
#    accept = 0
    for i in range(n_spins):
        flip()
#            accept += 1
#    return accept/N_site 

def thermalization():
    Total_E = []
    for i in range(thermal_steps):
        E = energy_total()
        ratio = oneMCstep()
        Total_E.append(E)
    return Total_E
 
def runMC():
    E_avr = 0
    for i in range(MC_step):
        E = energy_total()
        E_avr += E
        ratio = oneMCstep()
    E_avr = E_avr/MC_step
    
def get_vector(node):
    u = cos(theta[node])
    v = sin(theta[node])
    return u, v    

def plot_energy(Total_energy):
    steps = 1000
    plt.figure(2)
    plt.plot(list(range(len(Total_energy[:1000]))), Total_energy[0:1000])
    plt.ylabel('Energy')
    plt.xlabel('step')
    plt.title('Energy vs steps in thermalization')        
    plt.savefig('energy-steps.pdf')
    plt.close()
                           

def plot_graph():
    position = nx.spring_layout(G)
    X = []
    Y = []
    U = []
    V = []
    scale_vec = 0.08
    for e in position:
        X.append(position[e][0])
        Y.append(position[e][1])
        u, v = get_vector(e)
        u = scale_vec*u
        v = scale_vec*v
        U.append(u)
        V.append(v)
    plt.figure(1)
    ax = plt.gca()
    
    nx.draw(G, with_labels=True, pos=position)
    nx.draw_networkx_edge_labels(G, pos=position)
    ax.quiver(X, Y, U, V, scale = 1, angles='xy', scale_units='xy')
    plt.savefig('xy-graph.pdf')
    plt.close()

def main():
    initialize_rand()
    Total_energy = thermalization()
    runMC()
    plot_energy(Total_energy)
    plot_graph()
    
    
    
    
        
if __name__ == '__main__':
    main()   

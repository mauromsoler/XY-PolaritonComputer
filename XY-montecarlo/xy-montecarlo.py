import numpy as np
import scipy as sp
from math import pi
from random import *
from numpy import cos, sin
import matplotlib.pyplot as plt

Lx = 10
Ly = 10
Lz = 10

N_site = Lx*Ly*Lz

J_min = 1.6;
J_max = J_min + 0.03
J_step = 0.2
J = 1.6

MC_step = 500
thermal_steps = 2000

m_avr = 0
S_avr = 0

theta = np.zeros((Lx, Ly))

def initialize_rand():
    for y in range(Ly):
        for x in range(Lx):
            theta[x][y] = (2*pi)*random()
                 

def neighbours_periodic(x, y):
    x_p = 0 if x == Lx-1 else x+1
    x_m = Lx-1 if x == 0 else x-1
    y_p = 0 if y == Ly-1 else y+1
    y_m = Ly-1 if y == 0 else y-1     
    return x_p, x_m, y_p, y_m
    
    
def energy_in_site(x, y):
    x_p, x_m, y_p, y_m = neighbours_periodic(x, y)
    e = -J*(cos(theta[x][y] - theta[x_p][y]) 
            + cos(theta[x_m][y] - theta[x][y])
            + cos(theta[x][y] - theta[x][y_p])
            + cos(theta[x][y_m] - theta[x][y]) )
    return e
    
    
def energy_total():
    E = 0
    for y in range(Ly):
        for x in range(Lx):
            E += energy_in_site(x, y) 
    return E/2
    
    
def flip():
    x = np.random.randint(Lx)
    y = np.random.randint(Ly)
#    z = np.random.randint(Lz)
    
    e_save = energy_in_site(x, y)
    theta_save = theta[x][y]
    
    theta[x][y] += (2*pi)*random()
    e = energy_in_site(x, y)
    
    delta_e = e - e_save
    
    if delta_e < 0:
        return True
    else:
        flip_ratio = np.exp(-delta_e)
        if flip_ratio > random():
            return True
        else:
            theta[x][y] = theta_save
            return False
             
def oneMCstep():
#    accept = 0
    for i in range(N_site):
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
   

def get_vector(x,y):
    u = cos(theta[x][y])
    v = sin(theta[x][y])
    return u, v    

def plot_grid():  
    #X, Y = np.mgrid[0:Lx, 0:Ly]
    #U, V = get_vector(X, Y)
    #plt.quiver(X, Y, U, V)#, edgecolor='k', facecolor='None', linewidth=.5)
    #plt.show()  
    plt.figure(1)  
    X = list(range(Lx))
    Y = list(range(Ly))
    U = np.zeros((Lx, Ly))
    V = np.zeros((Lx, Ly))
    for x in X:
        for y in Y:
            U[x][y], V[x][y] = get_vector(x, y)
    plt.quiver(X, Y, U, V)
    plt.savefig('xy-model-end.pdf')
    plt.close() 
    
def plot_energy(Total_energy):
    steps = 1000
    plt.figure(2)
    plt.plot(list(range(len(Total_energy[:1000]))), Total_energy[0:1000])
    plt.ylabel('Energy')
    plt.xlabel('step')
    plt.title('Energy vs steps in thermalization')        
    plt.savefig('energy-steps.pdf')
    plt.close
                     
def main():
    initialize_rand()
    Total_energy = thermalization()
    runMC()
    plot_energy(Total_energy)
    plot_grid()
    
        
if __name__ == '__main__':
    main()    
          







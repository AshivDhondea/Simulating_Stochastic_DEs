## Euler_Maruyama_integration_AshivD.py
# Demonstration of Euler and Euler-Maruyama integration on the SHM problem.
# ------------------------- #
# Description:
# This project demonstrates numerical integration of the Simple Harmonic Motion DE
# using an Euler scheme. Numerical integration of the SHM DE perturbed by diagonal
# independent Wiener processes.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 04 July 2016
# Edits: 
# ------------------------- #
# Theoretical background
# 1. Bayesian Signal Processing, Sarkka. 2013.
# 2. ekfukf toolbox documentation
# ------------------------- #
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
#import sdeint
# f - frequency of SHM
f = 1.0; # [Hz]
omega2 = (2*np.pi*f)**2;

A = np.array([[0, 1.0],
              [ -omega2, 0]],dtype=np.float64);

def fnSHM(x,t):
    de = np.dot(A,x);
    return de

dt = f/1000;
num_samples =5*1/dt;
tspan = np.linspace(0.0, dt*num_samples, num_samples);

x0 = np.array([3.0, 0.0]);
X = np.zeros([2,len(tspan)],dtype=np.float64);
Y = np.zeros([2,len(tspan)],dtype=np.float64);
X[:,0] =x0;
P_0 = np.diag([1e-6,1e-6]);
Y[:,0] =x0 + np.random.multivariate_normal([0,0],np.sqrt(P_0));
# Dispersion matrix
L = np.array([[0.0,0.0],[0.0,1.0]],dtype=np.float64);
true_Qc =  np.diag([1.2,2.4]);
B = dt*true_Qc;

def G(x, t):
    return np.dot(L,B)

for index in range (0,len(tspan)-1):
    # Euler integration scheme.
    X[:,index+1] = X[:,index] + dt*fnSHM(X[:,index],tspan[index]);
    # Euler-Maruyama integration
    Y[:,index+1] = Y[:,index] + dt*fnSHM(Y[:,index],tspan[index])+ np.dot(L,np.random.multivariate_normal([0,0],B));
    #Y[:,index+1] = Y[:,index] + dt*fnSHM(Y[:,index],tspan[index])+ np.random.multivariate_normal([0,0],np.dot(L,B));

#result = sdeint.itoEuler(fnSHM, G, Y[:,0], tspan) # Euler-Maruyama

# Show results for the deterministic solution and the stochastic solution.
fig = plt.figure()
fig.suptitle('Euler integration on SHM DE')
plt.plot(tspan,X[0,:],'b.',label='Euler pos');
#plt.plot(tspan,Y[0,:],'bo',label='EM pos');
plt.plot(tspan,X[1,:],'r.',label='Euler vel');
#plt.plot(tspan,Y[1,:],'ro',label='EM vel');
plt.xlabel('t')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

## Compare results from my implementation with those from sdeint
fig = plt.figure()
fig.suptitle('Euler-Maruyama integration test')
plt.plot(tspan,Y[0,:],'b.',label='pos');
plt.plot(tspan,Y[1,:],'r.',label='vel');
#plt.plot(tspan,result[:,0],'g.',label='sdeint pos');
#plt.plot(tspan,result[:,1],'m.',label='sdeint vel');
plt.xlabel('t')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

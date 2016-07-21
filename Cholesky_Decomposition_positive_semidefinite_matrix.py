# -*- coding: cp1252 -*-
## Cholesky Decomposition of positive semidefinite matrix
# emulates schol.m of the ekfukf toolbox

##
# Ashiv Dhondea, RRSG, UCT.
# Created: 21 July 2016

import numpy as np
from numpy import linalg

# Positive semidefinite matrix from Page 167 in Tracking Filter Engineering. Norman Morrison 2013.
A = np.array([[1,-2],[-2,4]],dtype=np.float64);

def schol(A):
    ## Emulates schol.m of ekfukf toolbox.
    ## Description from schol.m
    ##    %SCHOL  Cholesky factorization for positive semidefinite matrices
    ##% Syntax:
    ##%   [L,def] = schol(A)
    ## % In:
    ##%   A - Symmetric pos.semi.def matrix to be factorized
    ##%
    ##% Out:
    ##%   L   - Lower triangular matrix such that A=L*L' if def>=0.
    ##%   def - Value 1,0,-1 denoting that A was positive definite,
    ##%         positive semidefinite or negative definite, respectively.
    ## % Copyright (C) 2006 Simo Särkkä
    n = np.shape(A)[0];
    L = np.zeros((n,n),dtype=np.float64);
    definite = 1;
    
    for i in range(0,n):
        for j in range(0,i+1):
            s = A[i,j];
            
            for k in range (0,j):
                s = s - L[i,k]*L[j,k];
            if j < i :
                if L[j,j] > np.finfo(np.float64).eps:
                    L[i,j] = s/L[j,j];
                else:
                    L[i,j] = 0;
            else:
                if (s < - np.finfo(np.float64).eps ):
                    s = 0;
                    definite = -1;
                elif (s < np.finfo(np.float64).eps):
                    s = 0;
                    definite = min(0,definite);
                    
                L[j,j] = np.sqrt(s);
                
    # if definite < 0, then negative definite
    # if definite == 0, then positive semidefinite
    # if definite == 1, then positive definite
    return L, definite

L, definite = schol(A);
print L
print definite

# Test if decomposition is correct
LLt = np.dot(L,L.T);
print LLt

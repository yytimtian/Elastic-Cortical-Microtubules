#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:37:38 2022

Testing the anchoring mechanism for the interacting simulation.
Copy & pasted functions from traj_change_test.py, tip_sim_on.py, deflect_distr.py.
Some changes to these for optimization and where the anchoring ends.


@author: tim
"""
import numpy as np
import scipy.integrate as ode
import scipy as sp
import random as rnd
from math import sqrt, log, pi
from time import time

def h_multi(x,y):
    '''
    Function for scipy's solve_bvp

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    return(np.vstack((y[1]/2, -4*np.cos(y[0])**3*np.sin(y[0]))))

def jacob(x,y):
    n = len(x)
    j = np.zeros((2,2,n))
    j[0,1,:] = 1/2
    j[1,0,:] = -2*(np.cos(4*y[0])+np.cos(2*y[0]))
    return(j)

def bcjac(ya,yb):
    return(np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]]))

def seg_sp_interval(L_seg, phi0, pt, ds):
    '''
    BVP solver using SciPy, returns solution and domain points.
    Domain points decided by ds on [0,pt].

    Parameters
    ----------
    L_seg : TYPE
        DESCRIPTION.
    phi0 : TYPE
        DESCRIPTION.
    pt : TYPE
        DESCRIPTION.
    ds: Float

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    epsilon=1e-4 #accuracy tolerance
    
    s = None
    
    N = np.ceil(pt/ds) #number of divisions till point of interest
    N = int(N)+1
    
    s = np.linspace(0,pt,N) #interval
    assert pt <= L_seg
    if pt != L_seg: #need to include endpoint to solve
        s = np.linspace(0,pt,N)
        s = np.append(s,L_seg) #grid points
    y_max = 2*pi 
    res = None
    m = 1 #parameter for initial guess value
    while y_max >= 2*pi: #iterate until guess results in ground state solution (y<=2pi)
        if m == 10:
            raise Exception('ODE iterative guess not converging well enough')
        y = np.zeros((2, s.size)) #initial guess
        if phi0 == 0: #zero case bifurcations
            if L_seg <= pi/(2*np.sqrt(2)):
                y[0] = 0 #let guess be zero
            else:
                y[0] = 1 #nonzero guess
        else: #need to be smart with initial guess, otherwise solver fails!
            if phi0 <= pi:
                y[0] = (pi/2 - phi0)*(1-2**(-m)) + phi0
            else:
                y[0] = (3*pi/2 - phi0)*(1-2**(-m)) + phi0
        
        def bc(ya,yb): #for BCs, don't know how to implement parameters within BCs
            return (np.array([ya[0]-phi0, yb[1]]))
        res = ode.solve_bvp(h_multi, bc, s, y, fun_jac = jacob, bc_jac = bcjac, tol=epsilon, max_nodes=50000) #solve
        y_max = res.y[0][-1]
        m += 0.5
    x, y, yd = None, None, None #initialize 
    if res.status != 0: #failed convergence, print why
        print(res.message, np.max(res.rms_residuals), phi0)
        print(L_seg, phi0, pt, ds)
        raise Exception('ODE solver failed')
    else:
        x = np.array([x for x in res.x if x<=pt]) #isolate domain of interest
        y = res.y[0,:len(x)]
        yd = res.y[1,:len(x)]
    return(x,y,yd)

def next_event(l0,del_l2):
    '''
    Determines next time

    Parameters
    ----------
    l0: current free tip length, as measured from previous anchor
    del_l2: v_g/k
    Returns
    -------
    Length of MT tip when anchoring happens, as measured from previous anchor

    '''
    u1 = rnd.random() #for next event time
    term = l0**2 - 2*del_l2*log(u1) #expression for tip length, squared
    return(sqrt(term))  

def sim_integrate(l_start,l_end,del_l2):
    '''
    Gillespie simulation of anchoring. For integrating phi.
    Only simulates lengths, phi calculation separate fn.
    Simulates up to AND INCLUDING the anchoring after l_end

    Parameters
    ----------
    l_start : Start length
    l_end : End length
    phi0 : Initial angles
    l0 : Initial tip length
    del_l2: v_g/k
    
    Returns
    -------
    L : Array of total MT lengths after each update
    pos : Array of anchors present
    phi : Array of corresponding angles

    '''
    #dynamical variables
    L = [l_start] #lengths at each update time
    pos = [0] #positions of anchors
    l0 = l_start
    tip_l = [0] #tip lengths prior to anchoring
    while pos[-1] < l_end:
        tip_lf = next_event(l0, del_l2) #tip length at time of update
        l_now  = L[-1]+tip_lf-l0 #length of MT now
        # if l_now > l_end: #use if we don't want to include anchoring beyond l_end
        #     break
        s = rnd.uniform(0, tip_lf) #anchoring point along tip, measured from last anchoring point
        L.append(l_now)
        pos.append(s+pos[-1]) #append values of latest anchor
        tip_l.append(tip_lf) #record the tip length
        l0 = tip_lf - s
    #add last point
    # tip_lf = l_end-pos[-1] #lenghth of tip from last anchoring
    # pos.append(l_end) #append values of latest anchor
    # tip_l.append(tip_lf) #new tip length is the difference
    # L.append(l_end)
    return(pos,tip_l,L)

def avg_phi_seg(L,pos,tip_l,phi0,ds): #average the angles along the traj
    M = len(L)-1
    phi0_avg = phi0 
    phi_avg = np.zeros(M) #array of average angles
    for i in range(M): #input average angles
        sol_avg = seg_sp_interval(tip_l[i+1], phi0_avg, pos[i+1]-pos[i], ds) #solve
        x_avg, y_avg = sol_avg[0], sol_avg[1]
        phi_bar = sp.integrate.trapz(y_avg,x_avg)/(pos[i+1]-pos[i]) #average
        phi_avg[i] = phi_bar
        phi0_avg = y_avg[-1] #anchoring angle to use next iter
    return(phi_avg)

def avg_phi_seg_secant(L,pos,tip_l,phi0,ds,secant_l): #average the angles, with specified secant lengths
    M = len(L)-1
    phi0_avg = phi0 
    phi_avg = []
    pos2 = [0] #positions of the secant segments -- these are the effective segments
    for i in range(M): #input average angles
        sol_avg = seg_sp_interval(tip_l[i+1], phi0_avg, pos[i+1]-pos[i], ds) #solve
        x_avg, y_avg = sol_avg[0], sol_avg[1]
        s_N = len(x_avg)
        secant_idx = [0]
        next_secant_pos = secant_l
        for j in range(s_N):
            new_seg = False #whether a new seg is calculated, tells us to integrate
            if x_avg[j] > next_secant_pos:
                assert j-1>0
                secant_idx.append(j-1)
                pos2.append(x_avg[j-1]+pos[i])
                next_secant_pos = x_avg[j-1] + secant_l
                new_seg = True
            elif j == s_N-1:
                secant_idx.append(j)
                pos2.append(x_avg[j]+pos[i])
                new_seg = True
            if new_seg: #time to integrate
                idx1 = secant_idx[-2] #start and end of integration interval
                idx2 = secant_idx[-1]
                phi_bar = ode.trapezoid(y_avg[idx1:idx2+1],x_avg[idx1:idx2+1])/(x_avg[idx2] - x_avg[idx1])
                phi_avg.append(phi_bar)
        phi0_avg = y_avg[-1] #anchoring angle to use next iter
    return(pos2, phi_avg)

def gen_path(phi0, l0, l_end, del_l2, ds=0.01): #generate traj using EL
    res = sim_integrate(l0,l_end,del_l2) #gives anchoring positions and times
    pos, tip_l, L = res[0], res[1], res[2] #extract results
    avg_phi = avg_phi_seg(L,pos,tip_l,phi0,ds) #get angles from EL
    new_l0 = tip_l[-1] - pos[-1] + pos[-2] #left over tip length
    N = len(pos) -1 #get distances bewtween seg
    dist = [] 
    for i in range(N):
        dist.append(pos[i+1]-pos[i])
    return(avg_phi,dist,new_l0)

def map_angle(phi): #map angle to be less than pi/2 for ODE solve purposes
    if phi >= pi/2 and phi < pi:
        phi = pi-phi
    elif phi >= pi and phi < 3*pi/2:
        phi = phi-pi
    elif phi >= 3*pi/2:
        phi = 2*pi - phi
    return(phi)

def adapt_secant(phi):
    '''
    Find secant length for MT shape approx. 

    Parameters
    ----------
    phi : Float
        Initial angle for ODE solve, mapped to [0,pi/2].

    Returns
    -------
    Secant length

    '''
    assert phi <= pi/2
    secant_l = None
    if phi <=0.3:
        secant_l = 0.05
    elif phi <0.62:
        secant_l = 0.05
    elif phi <= 1.1:
        secant_l = 0.1
    else:
        secant_l = 0.2
    return(secant_l)

def gen_path_secant(phi0, l0, l_end, del_l2, ds=0.01, adaptive = True): #generate traj using EL, fixed secant seg
    secant_l=None
    if adaptive: #increase accuracy depending on angle
        phi_test = map_angle(phi0) #map to <= pi/2
        secant_l = adapt_secant(phi_test)
    else:
        secant_l=0.2
    res = sim_integrate(l0,l_end,del_l2) #gives anchoring positions and times
    pos, tip_l, L = res[0], res[1], res[2] #extract results
    res = avg_phi_seg_secant(L,pos,tip_l,phi0,ds,secant_l)
    avg_phi, sec_pos = res[1], res[0]
    # avg_phi = avg_phi_seg_secant(L,pos,tip_l,phi0,ds,secant_l) #get angles from EL
    new_l0 = tip_l[-1] - pos[-1] + pos[-2] #left over tip length
    N = len(sec_pos) -1 #get distances bewtween seg
    dist = [] 
    for i in range(N):
        dist.append(sec_pos[i+1]-sec_pos[i])
    return(avg_phi,dist,new_l0)

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    timer = []
    for j in range(5):
        start = time()
        for i in range(1000):
            res = seg_sp_interval(1.2, .1, 0.5, 0.001)
        end = time()
        timer.append(end-start)
    print(timer)
    plt.plot(res[0],res[1])
    sys.exit()
    ds = 0.01
    start = time()
    rnd.seed(0)
    for k in range(30000):
        if k%1000==0:
            print(k)
        phi0 = .1#rnd.uniform(0,np.pi/2)
        # res = sim_integrate(0,.1,0.03529411764705882) #gives anchoring positions and times
        # pos, tip_l, L = res[0], res[1], res[2] #extract results
        # # print(pos)
        # avg_phi = avg_phi_seg(L,pos,tip_l,phi0,ds) #get angles from EL
        res = gen_path(phi0,0,0.1,0.1)
        # print(res[0],res[1])
    end = time()
    print('Time taken:',(end-start)/60)
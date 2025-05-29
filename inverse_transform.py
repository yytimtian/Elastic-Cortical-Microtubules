#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 19:00:46 2024

@author: tim
"""

import numpy as np
import scipy as sp
import scipy.integrate as integrate
import scipy.interpolate as interp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from math import pi, cos, sin
import random as rnd
import pickle
from os.path import exists

#nucleation parameters
ep = 0.89
f_left = 0.31
f_right = 0.31
th_b = 35/180*pi
N = 100000+1
vtol_nuc = pi/180

def ellipse_pdf(th,x):
    #PDF of ellipse nucleation
    const = (1-ep*ep)**(3/2)/(2*pi)/(f_left+f_right)
    left = (1/(1-ep*np.cos(th-th_b))**2)*f_left
    right = (1/(1-ep*np.cos(th+th_b))**2)*f_right
    return(const*(left+right))

def make_ellipse_table():
    '''
    Makes lookup table for sampling ellipse branching distribution

    Returns
    -------
    None.
    Makes ellipse_table.pickle lookup table in current directory.

    '''
    if not exists('./ellipse_table.pickle'):
        print('Making ./ellipse_table.pickle for LDD nucleation')
        theta = np.linspace(0,2*pi,N)
        pdf = ellipse_pdf(theta,0)
        
        # plt.plot(theta,pdf)
        cdf = integrate.cumulative_trapezoid(pdf,theta,initial=0)
        cdf_interp = interp.CubicHermiteSpline(theta, cdf, pdf)
        
        def cdf_interp2(x,y):
            return(cdf_interp.__call__(x)-y)
        
        cdf_range = np.linspace(0,1,N) #want uniform y-values
        theta_inverse = np.zeros(N) #for corresponding x values
        theta_inverse[-1] = 2*pi
        
        for i in range(1,N-1): #find x values for y values
            y = cdf_range[i]
            x = opt.root_scalar(cdf_interp2,args=(y),bracket=[0,2*pi], fprime=ellipse_pdf, xtol=1e-5)
            assert x.converged
            theta_inverse[i] = x.root
        dy = cdf_range[1]
        table ={
                'dtheta': dy,
                'theta_inverse': theta_inverse
                }
        
        file = 'ellipse_table'
        pickle_out = open(file+'.pickle','wb')
        pickle.dump(table,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        
# make_ellipse_table()

if __name__ == '__main__':
    table = None
    
    pick_in = open('ellipse_table.pickle','rb')
    table = pickle.load(pick_in)
    pick_in.close()
    
    dy_lookup = table['dtheta']
    theta_inverse = table['theta_inverse']

    def sample_angle():
        #generates elliptical angle for branched nucleation
        f_forward = 0.31
        f_backward = 0.07
        Y = rnd.uniform(0,1) #for choosing which mode of nucleation
        angle = 0
        if Y <= .31: #forward
            angle = 0
        elif Y <= .38:
            angle = pi #backward
        else: #branching
            #need to first import lookup table
            X = rnd.uniform(0,1)
            j = int(np.floor(X/dy_lookup))
            angle = theta_inverse[j]
            assert angle <= 2*pi and angle >= 0
            if angle < vtol_nuc or 2*pi-angle < vtol_nuc: #don't want almost parallel mts
                angle = 0
            elif abs(angle-pi) < vtol_nuc:
                angle = pi
        return(angle)
    # samples = np.zeros(M) #samples
    # for i in range(M): #inverse sampling
    #     X = rnd.uniform(0,1)
    #     assert X <= 2*pi and X >= 0
    #     j = int(np.floor(X/dy))
    #     samples[i] = theta_inverse[j]

    # theta = np.linspace(0,2*pi,100)
    # plt.hist(samples,bins=np.linspace(0,2*pi,1000), density = True)
    # plt.plot(theta, ellipse_pdf(theta,0))
    M = 1000000
    f_count = 0
    b_count = 0
    br_count = 0
    for i in range(M):
        angle = sample_angle()
        if angle == 0:
            f_count += 1
        elif angle == pi:
            b_count += 1
        else:
            br_count += 1
    print(f_count/M, b_count/M, br_count/M)
# -*- coding: utf-8 -*-
"""
For determining the parameters of the simulation

@author: tim
"""
import numpy as np
from math import pi, sin
deflect_on = True #whether to turn on delection (False <-> k_on = infty; ignores k_on, del_l2 below) ALWAYS ON, CHANGE vtol for this!! TODO:DELETE
global R, v_g, v_s, conv, f_gp, f_gs, f_pg, f_ps, f_sg, f_sg, f_sp, r_c, r_r, r_n, v_t, L_y0, l_grid

#checkpoint and data analysis info
final_hr = 10
time_snap = np.linspace(1,final_hr,10) #saving checkpoints
time_snap = np.append(time_snap,np.inf)
time_len = len(time_snap)-1
order_snap = np.linspace(0,final_hr, 10*time_len+1) #saving smaller info more frequently
order_snap = np.delete(order_snap, 0)
order_snap = np.append(order_snap,np.inf)
#
verbose = False
plot = False

no_bdl_id = False #whether we have spacing between bdls, if there's spacing then we cannot id bundles in a simple way
tread_bool = True
cat_on = True #catastrophe on or off
bdl_on = True
branch_stuff = True
pause_bool = True #only applies to arabidopsis case
pseudo_bdry = False #whether bdry acts as pseudo mts
#some more global settings
edge_cat = True #whether to turn on edge catastrophe at top+bottom
plant = 'by2' #arth = arabidopsis or by2 = tobacco by-2
vtol = pi/180 #2e-3 #tolerance for angle change WARNING NEED TO CHANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
vtol_nuc = pi/180 #tolerance for branched nucleation
tub_dim = .008 #tubulin dimer length um
ep = 0 #parameter for angle-dependant catastrophe
angle_dep_cat = False #whether there is angle dep
#LDD parameters
LDD_bool = False
R_meta = 1.5 #meta traj radius, micron
r_ins = 0.0045 #nuc complex insertion rate, 1/micron^2/s
r_u = 0.002 #nuc complex unbound rate, 1/s
D_meta = 0.013 #nuc complex diffusion, micron^2/s
accept_MT = 0.24 #prob of MT-based nucleation
accept_unbound = 0.02 #prob of unbound nucleation
r_ins *= accept_MT #rescaling for efficiency
accept_unbound = accept_unbound/accept_MT

if no_bdl_id:
    assert vtol == 2*pi #no bdl case only works for geodesics
    assert not angle_dep_cat
    
if plant == 'arth':
    #For testing purposes only. Uses 3 state model. Coded before LDD mode, I need to check if it's still consistent.
    #reference parameters - set to 1
    R = 5 #radius in um
    v_g = 6.5/60 #0.08 #growth speed in um/s
    v_s = 12/60
    tub = tub_dim/R
    #time conversion
    conv = R/v_g
    #other parameters w/ dim
    #Jun's 3 state model
    f_gp = 0.38/60
    f_gs = 1.59/60
    f_pg = 1.4/60
    f_ps = 0.7/60
    f_sg = 1.99/60
    f_sp = 0.44/60
    
    if not pause_bool:
        r_c = (f_gp*f_ps+f_gs*f_pg+f_gs*f_ps)/(f_gp+f_pg+f_ps)
        r_r = (f_pg*f_sg+f_pg*f_sp+f_ps*f_sg)/(f_sp+f_pg+f_ps)
        v_g = (f_pg+f_ps)/(f_gp+f_pg+f_ps)*v_g
        v_s = (f_pg+f_ps)/(f_sp+f_pg+f_ps)*v_s
        
        r_c0 = r_c/60 #0.0045/100 #cat rate 1/s
        r_r0 = r_r/60 #0.007/100 #rescue rate 1/s
        r_c = r_c0*R/v_g #scaled cat rate
        r_r = r_r0*R/v_g #scaled rescue rate
    
    r_n0 = 10/60 #0.001/100 #nucleation rate 1/(s*um^2)    
    v_s /= v_g
    r_n = r_n0*R**3/v_g #nucleation rate
    
    L_y0 = 2*R*pi #height in um (really z axis in R^3 but kept as kept as y in pre-image to be consistent w/ code)
    # l_scale = 2*(v_g-0.53)**2*(2*v_g+0.53)/(10*v_g*(v_g+2*v_g))
    # l_scale = l_scale**(1/3)
    l_grid = 1 #l_scale #ref grid length in um
    #other parameters rescaled
    # v_s = 2 #scaled speed, growth is set to 1
    v_t = .53/6.5 #0.125 #scaled minus end shrinkage
else:
    #reference parameters - set to 1
    pause_bool = False
    R = 20 #radius in um
    v_g = 0.08 #growth speed in um/s
    tub = tub_dim/R
    #time conversion
    conv = R/v_g
    
    r_c0 = 0.0045 #cat rate 1/s
    r_r0 = 0.007 #rescue rate 1/s
    r_c = r_c0*R/v_g #scaled cat rate
    r_r = r_r0*R/v_g #scaled rescue rate
    
    r_n0 = 0.001 #nucleation rate 1/(s*um^2)    
    r_n = r_n0*R**3/v_g #nucleation rate
    
    L_y0 = 2*pi*R #height in um (really z axis in R^3 but kept as kept as y in pre-image to be consistent w/ code)
    # l_scale = 2*(v_g-0.53)**2*(2*v_g+0.53)/(10*v_g*(v_g+2*v_g))
    # l_scale = l_scale**(1/3)
    l_grid = 5 #l_scale #ref grid length in um
    #other parameters rescaled
    v_s = 2 #scaled speed, growth is set to 1
    v_t = 0.125 #scaled minus end shrinkage
    #for angle-dep cat
    angle_part_n = 4 #number of angle partitions
    angle_part = np.linspace(0,pi/2,angle_part_n+1) #catastrophe angle partition
    # cat_scale = 2 #scaling factor for this particular cat model
    r_part = r_c*(1+0.2*np.sin(angle_part))
    if ep == 0:
        angle_dep_cat = False
    if LDD_bool:
        r_ins = r_ins*R**3/v_g #non-dim
        R_meta = R_meta/R
        r_u = r_u*R/v_g
        D_meta = D_meta/R/v_g
        r_n = r_ins #nucleation rate is r_ins
    #order of dict is important!!!
    rate_dict = {
        'r_n': r_n,
        'r_r': r_r,
        'r_c': r_c
        }

    
kon_min = 0.34 #1/(min*um)
kon0 = kon_min/60 #1/(s*um)
kon = kon0*R**2/v_g #anchoring rate
del_l2 = 1/kon #anchoring length scale
L_y = L_y0/R
L_x = 2*pi
dr = 2.5e-2/R #bundling distance, if used
dr_tol = dr/sin(pi/180) #tolerance for step back distance
'--------------------------------------END OF PARAMETER SETTINGS-----------------------------------------'
#split into grids
xdomain = [0,L_x] #scaled domain
ydomain = [0,L_y]
n = int(np.ceil(L_y*R/l_grid)) #number of grid divisions for each axis
m = int(np.ceil(L_x*R/l_grid))
#determine grid locations
grid_l, grid_w = n,m #number of regions length and width wise
grid =[grid_w,grid_l] #lxw of thread
dx = (xdomain[1]-xdomain[0])/grid_w #measurements of each grid dimension
dy = (ydomain[1]-ydomain[0])/grid_l
x_interval = [] #intervals for each region
y_interval = []
for i in range(grid_l*grid_w): #fill in intervals
    col = i%grid_w #column Number
    row = np.floor(i/grid_w)
    x1, x2 = col*dx, (col+1)*dx
    y1, y2 = row*dy, (row+1)*dy
    x_interval.append([x1,x2])
    y_interval.append([y1,y2])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:41:58 2021

@author: tim
"""
import matplotlib.pyplot as plt
import numpy as np
from math import sin,cos,pi,sqrt,atan
from parameters import v_s, xdomain, ydomain, x_interval, y_interval, grid_l, grid_w, L_y, conv
import copy

def plot_snap(mt_list, t,k,dest='./plots3/',save=True, region=False, nodes=True, plot_region = -1, show = False, seed=-1, color=False, result=None, plot_pts = []):
    red, blue, cyan, green = 'red', 'blue', 'cyan', 'green'
    if not color:
        red, blue, cyan, green = 'limegreen', 'limegreen', 'limegreen', 'limegreen'
    l = len(mt_list)
    # cmap = plt.cm.get_cmap('winter')
    # bdl_mts = [len(b.mts) for b in bdl_list]
    # norm = mt.colors.Normalize(vmin=min(bdl_mts), vmax=max(bdl_mts))
    for i in range(l):
        coord = mt_list[i].seg #get coordinates of segments
        x_seg, y_seg = [p[0] for p in coord], [p[1] for p in coord] #historical segments
        x_segp, y_segp = [p[0] for p in coord], [p[1] for p in coord] #current segments
        th = mt_list[i].angle[-1] #angle
        t_i = mt_list[i].update_t[-1] #last update time
        mt1 = mt_list[i]
        #treadmilling business
        d_tr1 = 0 #treadmilling dist
        tr_pos1 = 0 #-ve end position, as measured from last vertex
        seg_start1 = 0 #the segment to start comparisons on, can skip some due to treadmilling
        mt1 = mt_list[i]
        if mt1.exist:
            # color = norm(bdl_mts[mt1.bdl])
            # color = cmap(color)
            # red, blue, cyan, green = color, color, color, color
            if mt1.tread: #calculate treadmilling seg and pos for mt2
                old_dist1 = mt1.seg_dist
                l1 = len(x_seg)
                d_tr1 = (t-mt1.tread_t)*mt1.vt
                tr_pos1 = (t-mt1.tread_t)*mt1.vt
                seg_dist = 0 #total dist of segs considered so far
                seg_start1 = 0
                for m in range(l1-1):
                    seg_dist += old_dist1[m]
                    if d_tr1 <= seg_dist:
                        break
                    else:
                        seg_start1 += 1
                        tr_pos1 -= old_dist1[m]
                x_segp, y_segp = x_segp[seg_start1:], y_segp[seg_start1:] #the segments that still exist on negaive end
                x_segp[0] += cos(mt1.angle[seg_start1])*tr_pos1
                y_segp[0] += sin(mt1.angle[seg_start1])*tr_pos1
        if mt_list[i].exist: #if it exists, plot segments
            if nodes: #nodes
                plt.scatter(x_seg,y_seg,s=.5, color='black') #scatter of all segment points
        if t_i == t: #if it's been updated at this time
            if mt_list[i].exist:
                # if mt_list[i].grow or mt_list[i].hit_bdry: #growing or stationary
                # print(mt1.number, x_segp, y_segp,k)
                x_i = x_segp[:-1] #all but most recent update
                y_i = y_segp[:-1]
                x_f = x_segp[-2:] #updated segment
                y_f = y_segp[-2:]
                plt.plot(x_i,y_i,color=green,linewidth=0.5, alpha = 0.5)
                if len(x_f)>1:
                    plt.plot(x_f,y_f,color=red,linewidth=0.5, label='MT event', alpha = 0.5)
                else:
                    plt.plot(x_segp,y_segp,color=green,linewidth=0.5, alpha = 0.5) #started to shrink
            else:
                if nodes: #plot the vertices
                    plt.scatter([x_seg[0]],[y_seg[0]], s=.5,color='purple') #yellow points are dissapeared MTs
        else: #if it's continuing to grow at this point in time
            if mt_list[i].exist:
                if mt_list[i].grow: #growing
                    x2 = x_seg[-1] + cos(th)*(t-t_i)
                    y2 = y_seg[-1] + sin(th)*(t-t_i)
                    # print(mt_list[i].number, x2,y2,k)
                    if not (x2 <= xdomain[1] and x2 >=xdomain[0] and y2 <= ydomain[1] and y2 >=ydomain[0]):
                        print('MT out of domain!', i)
                        assert False
                    x_f = [x_segp[-1], x2] #where it has grown to
                    y_f = [y_segp[-1], y2]
                    plt.plot(x_segp,y_segp,color=green,linewidth=0.5, alpha = 0.5)
                    plt.plot(x_f,y_f, color=blue, linewidth=0.5, alpha = 0.5)
                elif mt_list[i].hit_bdry: #stationary
                    plt.plot(x_segp,y_segp,color=green,linewidth=0.5, alpha = 0.5)
                else: #shrinking
                    assert not mt_list[i].grow
                    cumsum = np.append([0],np.cumsum(mt_list[i].seg_dist)) #cumulative sums
                    d = (t- t_i)*v_s #distance traversed
                    left_over = cumsum[-1] - d
                    j = 0
                    while left_over > cumsum[j]: #find index within cumsum
                        j+=1
                        if j > len(cumsum)-1:
                            break
                    j -= 1
                    delta = left_over - cumsum[j] #component sticking out from segment
                    x_i = x_seg[seg_start1:j+1] #untouched points in shrinkage
                    y_i = y_seg[seg_start1:j+1]
                    x_i[0], y_i[0] = x_segp[0], y_segp[0] #starting pt may be different due to treadmilling
                    th = mt_list[i].angle[j]
                    x_f = [x_i[-1], x_seg[j] + cos(th)*delta] #where it has grown to
                    y_f = [y_i[-1], y_seg[j] + sin(th)*delta]
                    plt.plot(x_i,y_i,color=green,linewidth=0.5, alpha = 0.5)
                    plt.plot(x_f,y_f,color=cyan,linewidth=0.5, alpha = 0.5)
            else:
                0
                # if nodes:
                #     plt.scatter([x_seg[0]],[y_seg[0]],s=0.5,color='purple')
    if region: #if we want to plot the regions
        for i in range(grid_w-1):
            x = [x_interval[i][1], x_interval[i][1]]
            y = [ydomain[0], ydomain[1]]
            plt.plot(x,y,color='gray', linewidth=0.5, linestyle='--')
        for j in range(grid_l-1):
            idx = j*grid_w
            y = [y_interval[idx][1], y_interval[idx][1]]
            x = [xdomain[0], xdomain[1]]
            plt.plot(x,y,color='gray', linewidth=0.5, linestyle='--')
    dr = 0 #offset for troubleshooting
    L_x, L_y = xdomain[1], ydomain[1]
    if type(plot_region) is int: #if default or only one region to plot
        if plot_region >= 0:
            x_domain, y_domain = x_interval[plot_region], y_interval[plot_region] #find intervals
            plt.xlim([x_domain[0]-dr, x_domain[1]+dr])
            plt.ylim([y_domain[0]-dr, y_domain[1]+dr])
        else:
            plt.xlim([-dr,L_x+dr])#xdomain)
            plt.ylim([-dr,L_y+dr])#ydomain)
    else: #if multiple regions, it's a tuple
        x_domain, y_domain = [], []
        for r in plot_region:
            x_domain += x_interval[r]
            y_domain += y_interval[r]
        xmin, xmax = np.min(x_domain), np.max(x_domain) #plot part including all regions
        ymin, ymax = np.min(y_domain), np.max(y_domain)
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
    if len(plot_pts) > 0:
        for pt in plot_pts:
            if pt != None:
                sp_pt = pt #[3.415183577039977, 1.2105502560939234]
                plt.scatter(sp_pt[0],sp_pt[1],c='None',edgecolors='C1')
    # Dx = 0.0005
    # plt.xlim([0.97,1.0])
    # plt.ylim([2,2.02])
    # xx = np.linspace(sp_pt[0]-0.999*Dx,sp_pt[0]+0.999*Dx,100)
    # yy1 = -np.sqrt(Dx**2-(xx-sp_pt[0])**2)+sp_pt[1]
    # yy2 = np.sqrt(Dx**2-(xx-sp_pt[0])**2)+sp_pt[1]
    # plt.plot(xx,yy1)
    # plt.plot(xx,yy2)
    # plt.xlabel(r'$\theta$',fontsize=30)
    # plt.ylabel(r'$z$',fontsize=30)
    ax = plt.gca()
    xmax, ymax = xdomain[-1], ydomain[-1]
    # ax.set_yticks([0,ymax/4, ymax/2, 3*ymax/4, ymax])
    # ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
    # ax.set_yticklabels([r'$0$',r'$H/4$',r'$H/2$',r'$3H/4$',r'$H$'],fontsize=20)
    # ax.set_xticks([0,xmax/4, xmax/2, 3*xmax/4, xmax])
    # ax.set_xticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
    # ax.set_xticklabels([r'$0$',r'$L/4$',r'$L/2$',r'$3L/4$',r'$L$'],fontsize=20)
    # if seed == -1:
    #     plt.title('Time {} hr, iteration {}'.format(t*conv/(60*60),k))
    # else:
    #     plt.title('Time {} hr, seed {}'.format(t*conv/(60*60),seed))
    if result != None:
        plt.title('Time= {} hr, i {}, {} \n {}'.format(t,k,result.policy,result.pt),fontsize=8)
    plt.gca().set_aspect('equal',adjustable='box')
    figure = plt.gcf()
    figure.set_size_inches(10, 10)
    name = '_plot'
    if save:
        orient = 'portrait'
        if L_y > L_x:
            orient = 'landscape'
        plt.tight_layout()
        plt.savefig(dest+name+'.pdf',format='pdf', orientation = orient)
    if show:
        plt.show()
    plt.clf()
    return(0)

def U_tens(phi,theta,l): #U tensor
    beta = pi/2-theta
    alpha = phi-pi/2
    ca, sa = cos(alpha), sin(alpha)#-sin(phi), cos(phi) #alpha = pi/2 + phi so sin, cos are changed
    cb, sb = cos(beta), sin(beta)#sin(theta), cos(theta) #beta = pi/2 - theta
    U_xx = l*ca**2 #U tensor
    U_xy = ca*(-sb + sin(beta + l*sa))
    U_xz = ca*(-cb + cos(beta+l*sa))
    U_yy = sa*(2*l*sa-sin(2*beta)+sin(2*beta+2*l*sa))/4
    U_yz = sa*(-cb**2+cos(beta+l*sa)**2)/2
    U_zz = sa*(2*l*sa+sin(2*beta)-sin(2*beta+2*l*sa))/4
    U = np.zeros((3,3))
    U[0,0] = U_xx
    U[0,1], U[1,0] = U_xy, U_xy
    U[0,2], U[2,0] = U_xz, U_xz
    U[1,1] = U_yy
    U[1,2], U[2,1] = U_yz, U_yz
    U[2,2] = U_zz
    return(U)

def order_param(mt_list,t):
    l = len(mt_list)
    U = np.zeros((3,3)) #order tensor
    mt_len = 0
    for i in range(l):
        mt = mt_list[i]
        t_i = mt.update_t[-1] #last update time
        if mt.exist:
            if mt.grow: #growing
                seg_l = len(mt.seg_dist) #number of segments
                for i in range(seg_l): #fixed segments
                    phi, theta, l = mt.angle[i], mt.seg[i][0], mt.seg_dist[i]
                    U_temp = U_tens(phi,theta,l)
                    U += U_temp
                    mt_len += l
                phi, theta, l = mt.angle[-1], mt.seg[-1][0], t-t_i #growing tip
                U_temp = U_tens(phi,theta,l)
                U += U_temp
                mt_len += l
            elif mt.hit_bdry: #stationary
                seg_l = len(mt.seg_dist) #number of segments
                for i in range(seg_l): #fixed segments
                    phi, theta, l = mt.angle[i], mt.seg[i][0], mt.seg_dist[i]
                    U_temp = U_tens(phi,theta,l)
                    U += U_temp
                    mt_len += l
            else: #shrinking
                assert not mt.grow
                cumsum = np.append([0],np.cumsum(mt.seg_dist)) #cumulative sums
                d = (t- t_i)*v_s #distance traversed
                left_over = cumsum[-1] - d
                j = 0
                while left_over > cumsum[j]: #find index within cumsum
                    j+=1
                    if j > len(cumsum)-1:
                        break
                j -= 1
                delta = left_over - cumsum[j] #component sticking out from segment
                new_seg = mt.seg[:j+1]
                seg_l = len(new_seg)
                for i in range(seg_l): #fixed segments
                    phi, theta, l = mt.angle[i], mt.seg[i][0], mt.seg_dist[i]
                    U_temp = U_tens(phi,theta,l)
                    U += U_temp
                    mt_len += l
                phi, theta, l = mt.angle[j], mt.seg[j][0], delta #shrinking tip
                U_temp = U_tens(phi,theta,l)
                U += U_temp
                mt_len += l
    U /= mt_len
    T = np.zeros((3,3)) #correction matrix
    den = 2*L_y + 2
    T[0,0], T[1,1], T[2,2] = L_y/den, (L_y+2)/(2*den), (L_y+2)/(2*den)
    Q = U - T
    res = np.linalg.eig(Q)
    idx = np.argmin(res[0])
    vmin, lamb = res[1][:,idx], res[0][idx] #eigenstuff
    coeff = max(L_y, (L_y+2)/2)
    coeff /= 2*L_y+2
    ex = np.array([1,0,0])
    th = np.arccos(abs(np.sum(ex*vmin)))
    assert th != pi
    return(vmin, -lamb/coeff, th)

def order_hist(mt_list,t):
    l = len(mt_list)
    mt_len = 0
    angles = []
    leng = []
    theta = []
    for i in range(l):
        mt = mt_list[i]
        t_i = mt.update_t[-1] #last update time
        tr_pos1 = 0 #set here to use when needed
        seg_start1 = 0
        if mt.exist:
            mt_angle, mt_seg, mt_seg_dist = copy.deepcopy(mt.angle), copy.deepcopy(mt.seg), copy.deepcopy(mt.seg_dist) #will need to modify these to get order params
            if mt.tread: #calculate treadmilling seg and pos for mt2
                old_dist1 = mt.seg_dist
                l1 = len(mt.seg)
                d_tr1 = (t-mt.tread_t)*mt.vt
                tr_pos1 = (t-mt.tread_t)*mt.vt
                seg_dist = 0 #total dist of segs considered so far
                seg_start1 = 0
                for m in range(l1-1):
                    seg_dist += old_dist1[m]
                    if d_tr1 <= seg_dist:
                        break
                    else:
                        seg_start1 += 1
                        tr_pos1 -= old_dist1[m]
                if mt.grow or mt.hit_bdry: #can calculate ahead for these cases
                    mt_angle = mt_angle[seg_start1:] #the segments that still exist on negative end
                    mt_seg = mt_seg[seg_start1:]
                    mt_seg[0][0] = mt_seg[0][0] + cos(mt.angle[seg_start1])*tr_pos1
                    mt_seg[0][1] = mt_seg[0][1] + sin(mt.angle[seg_start1])*tr_pos1
                    mt_seg_dist = mt_seg_dist[seg_start1:]
                    if len(mt_seg_dist) >= 1: #physical minus end point
                        mt_seg_dist[0] = sqrt((mt_seg[0][0]-mt_seg[1][0])**2 + (mt_seg[0][1]-mt_seg[1][1])**2)
            if mt.grow: #growing
                seg_l = len(mt_seg_dist) #number of segments
                for i in range(seg_l): #fixed segments
                    phi, th, l = mt_angle[i], mt_seg[i][0], mt_seg_dist[i]
                    angles.append(phi)
                    leng.append(l)
                    theta.append(th)
                    mt_len += l
                phi, th, l = mt_angle[-1], mt_seg[-1][0], t-t_i #growing tip
                if len(mt_seg_dist) == 0: #if treadmilling occurs and it's up to a single seg left, subtract the treaded amount
                    l -= tr_pos1
                angles.append(phi)
                leng.append(l)
                theta.append(th)
                mt_len += l
            elif mt.hit_bdry: #stationary
                seg_l = len(mt_seg_dist) #number of segments
                for i in range(seg_l): #fixed segments
                    phi, th, l = mt_angle[i], mt_seg[i][0], mt_seg_dist[i]
                    angles.append(phi)
                    leng.append(l)
                    theta.append(th)
                    mt_len += l
            else: #shrinking
                assert not mt.grow
                cumsum = np.append([0],np.cumsum(mt.seg_dist)) #cumulative sums
                d = (t- t_i)*v_s #distance traversed
                left_over = cumsum[-1] - d
                j = 0
                while left_over > cumsum[j]: #find index within cumsum
                    j+=1
                    if j > len(cumsum)-1:
                        break
                j -= 1
                delta = left_over - cumsum[j] #component sticking out from segment
                mt_angle = mt_angle[seg_start1:j+1] #the segments that still exist on negative end
                mt_seg = mt_seg[seg_start1:j+1]
                mt_seg[0][0] = mt_seg[0][0] + cos(mt.angle[seg_start1])*tr_pos1 #minus end point
                mt_seg[0][1] = mt_seg[0][1] + sin(mt.angle[seg_start1])*tr_pos1
                mt_seg_dist = mt_seg_dist[seg_start1:j]
                if len(mt_seg_dist) >= 1:
                    mt_seg_dist[0] = sqrt((mt_seg[0][0]-mt_seg[1][0])**2 + (mt_seg[0][1]-mt_seg[1][1])**2)
                seg_l = len(mt_seg_dist)

                for i in range(seg_l): #fixed segments
                    phi, th, l = mt_angle[i], mt_seg[i][0], mt_seg_dist[i]
                    angles.append(phi)
                    leng.append(l)
                    theta.append(th)
                    mt_len += l
                phi, th, l = mt_angle[-1], mt_seg[-1][0], delta #shrinking tip
                if len(mt_seg_dist) == 0: #if treadmilling occurs and it's up to a single seg left, subtract the treaded amount
                    l -= tr_pos1
                angles.append(phi)
                leng.append(l)
                theta.append(th)
                mt_len += l
    # for j in range(len(angles)): #only care about angles between 0 and pi -- not needed?
    #     if angles[j] >= pi/2 and angles[j] < pi:
    #         angles[j] = angles[j] - pi
    #     if angles[j] < 3*pi/2 and angles[j] >= pi:
    #         angles[j] = angles[j] - pi
    return(angles,leng,mt_len,theta)

def s2(geom):
    angles, leng, mt_len = np.array(geom[0]), np.array(geom[1]), geom[2]
    num = (np.sum(leng*np.cos(2*angles)))**2 + (np.sum(leng*np.sin(2*angles)))**2
    num = sqrt(num)
    s2 = num/mt_len
    arg = np.sum(leng*np.sin(2*angles))/mt_len
    arg /= np.sum(leng*np.cos(2*angles))/mt_len+s2
    theta = atan(arg)
    # theta1 = atan(np.sum(leng*np.cos(2*angles))/np.sum(leng*np.sin(2*angles)))
    return(s2,theta)

def s2_new(geom):
    angles, leng, mt_len = np.array(geom[0]), np.array(geom[1]), geom[2]
    arg = (np.sum(leng*np.sin(2*angles)))/(np.sum(leng*np.cos(2*angles)))
    omega = atan(arg)/2
    num = (np.sum(leng*np.cos(angles-omega)**2))-(np.sum(leng*np.sin(angles-omega)**2))
    s2 = num/mt_len
    return(s2,omega)

def plot_hist_mtlist(mt_list, t):
    ax = None
    hist_res = order_hist(mt_list,t)
    omega_indiv = s2(hist_res)
    s2_indiv, omega_indiv = omega_indiv[0], omega_indiv[1]
    for i in range(len(hist_res[0])):
        hist_res[0][i] -= omega_indiv
        angle = hist_res[0][i]
        if angle >= pi/2 and angle <= 3*pi/2: #convert to be in [-pi/2,pi/2]
            hist_res[0][i] -= pi
        if angle > 3*pi/2:
            hist_res[0][i] -= 2*pi
        # angle = hist_res[0][i] - omega_indiv #shift by dominant angle
    for i in range(len(hist_res[0])):
        angle = hist_res[0][i]
        if angle < -pi/2:
            hist_res[0][i] += pi
    binss = np.linspace(-pi/2,pi/2,60)
    ax = plt.axes()
    plt.hist(hist_res[0],binss,weights=hist_res[1],density=True,edgecolor = "black")
    ax.set_xticks([-pi/2, -pi/4, 0, pi/4,pi/2])
    ax.set_xticklabels([r'$-\pi/2$',r'$-\pi/4$', r'$0$',r'$\pi/4$',r'$\pi/2$'])
    plt.ylabel('Weighted Frequency', fontsize=14)
    plt.xlabel(r'$\varphi - \Omega$', fontsize=14)
    plt.title(r'$\Omega=${:.2f}, $S_2=${:.2f}'.format(omega_indiv,s2_indiv),fontsize=14)
    # plt.savefig('seed'+str(k)+'_10hr_hist.pdf', format='pdf',bbox_inches='tight')
    plt.show()
    plt.clf()

def plot_hist_geom(geom):
    angles, leng = geom[0], geom[1]
    binss = np.linspace(-pi/2,pi/2,60)
    ax = plt.axes()
    plt.hist(angles,binss,weights=leng,density=True,edgecolor = "black")
    ax.set_xticks([-pi/2, -pi/4, 0, pi/4,pi/2])
    ax.set_xticklabels([r'$-\pi/2$',r'$-\pi/4$', r'$0$',r'$\pi/4$',r'$\pi/2$'])
    plt.ylabel('Weighted Frequency', fontsize=14)
    plt.xlabel(r'$\varphi - \Omega$', fontsize=14)
    # plt.title(r'$\Omega=${:.2f}, $S_2=${:.2f}'.format(omega_indiv,s2_indiv),fontsize=14)
    # plt.savefig('seed'+str(k)+'_10hr_hist.pdf', format='pdf',bbox_inches='tight')
    plt.show()
    plt.clf()
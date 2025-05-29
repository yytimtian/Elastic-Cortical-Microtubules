#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:14:24 2021

@author: tim

For functions used in comparing two points
"""
import numpy as np
from math import atan, sin, cos, sqrt, pi, tan, log
from parameters import v_s, v_t, xdomain, ydomain, x_interval, y_interval, dx, dy, grid_w, deflect_on, vtol, tub, grid_w, grid_l
from parameters import conv, no_bdl_id, plant, angle_dep_cat, no_bdl_id, vtol_nuc, tub_dim, R_meta
if plant == 'arth':
    0
else:
    from parameters import r_c, ep, angle_dep_cat
from sim_anchoring_alg_test import gen_path, gen_path_secant
import copy
import warnings
import random as rnd
import pickle
# import sys

def check_fixed_region(l_idx1,l_idx2, mt_list, t, r):
    # l_idx1, l_idx2 = idx1,idx2#mt_to_l(mt_list,idx1), mt_to_l(mt_list, idx2) #get list indices
    mt1, mt2 = mt_list[l_idx1], mt_list[l_idx2] #get corresponding MTs
    result = False
    grow1, grow2 = mt1.grow, mt2.grow
    if grow1==False and grow2==False: #if both are shrinking
        0
    else:
        if mt1.region == mt2.region:
            result = True
    return(result)

def sort_and_find_events(event_list, regions):
    for k in regions: #sort according to regions where MTs changed
        event_list[k].sort(key=lambda x: x.t)
    comparison_list = [elists for elists in enumerate(event_list) if len(elists[1]) > 0] #get non-zero length event lists
    new_region, new_event_list = min(comparison_list, key=lambda x: x[1][0].t) #
    return(new_region, new_event_list[0])

class event:
    """
    Class for events
    """
    def __init__(self,mt1,mt2,t,pt,plcy,col_idx = None,calc_t = None):
        '''
        Initialize event

        Parameters
        ----------
        mt1 : mt class
            1st MT.
        mt2 : mt class
            2nd MT.
        t : Float
            Event time.
        pt : List of floats
            Event pt.
        plcy : String
            Policy.
        col_idx : Int, optional
            Vertex index of barrier MT. The default is None.
        calc_t : Float, optional
            Time of event calculation. The default is None.

        Returns
        -------
        None.

        '''
        self.mt1_n = mt1.number #mt numbers
        self.mt2_n = mt2.number
        self.bdl1 = None
        self.bdl2 = None
        if plcy != 'nucleate':
            self.bdl1 = mt1.bdl
            self.bdl2 = mt2.bdl
        self.t = t #event time
        self.pt = pt #event point [x,y]
        self.policy = plcy #event policy '1hit2' etc.
        self.mt_idx = col_idx #segment idx of collition, if there is there is collision
        self.calc_t = calc_t
        self.prev = plcy
        self.prev_pt = pt
        if mt1.bdl != None:
            if mt1.bdl == mt2.bdl:
                if self.policy != '1catch2' and self.policy != '1catch2_m':
                    print(mt1.number,mt2.number,mt2.bdl, self.policy)
                    assert mt1.bdl != mt2.bdl

class mt:
    """
    Class for indiv. MTs
    """
    def __init__(self,number,free=False):
        '''
        Initialize MT. Could add more to this later.

        Parameters
        ----------
        number : Int
            MT identity.
        free : Bool, optional
            Whether MT is free to bend. The default is False.

        Returns
        -------
        None.

        '''
        self.number = number
        self.pseudo_bdry = False
        self.bdl = None #bundle number if exists
        self.exist = True #whether MT has disappeared due to shrinking
        self.seg = [] #nx2 array to store previous segment points
        self.traj = [] #trajectories each seg is on
        self.seg_dist = [] #n-1 array to store distance. ith is dist. from seg[i] to seg[i+1]
        self.angle = []# n array angles of trajectories corresponding to seg points

        self.update_t = [0.0] #time of last segment point update

        self.grow = True #bool for growth or shrink

        self.hit_bdry = False #bool for whether it hit bdry
        self.ext_mt = None #int for MT index of continued MT

        self.from_bdry = False #bool for whether it is extension from an MT which hit bdry
        self.from_bdry_2 = False #whether it's actually from bdry
        self.prev_mt = None #MT index from which it is an extension of
        self.prev_t = None #previous physical trajectory change time if it's a continuation
        self.tip_l = None #tip length
        self.region = None #region MT belongs to
        self.free = free #whether it's on a bdl or not

        self.tread = False#tread_bool #treadmilling
        self.tread_t = -1
        self.vt = v_t
        #pre-generated mt path
        self.path_dist = []
        self.path_angle = []
        self.tip_lf = 0 #final tip length from generation, may use for generating more
        self.path_idx = 0 #index of next path to choose
        #info on length if free:
        self.total_len = 0
        self.init_angle = None
    def add_vertex(self, pt, th, t, resolve, traj_id = None, grow = True, double_check = False):
        '''
        Add new vertex info.

        Parameters
        ----------
        pt : Array of floats
            Vertex pt.
        th : Float
            Traj angle.
        t : Float
            Time of update.
        resolve : String
            Resultant event.
        traj_id : Int, optional
            Traj ID. The default is None.
        grow : Bool, optional
            Whether MT is growing. If not, adding vertex is optional. The default is True.

        Returns
        -------
        None.

        '''
        if self.grow:
            self.seg.append(pt) #2D point
            self.update_t.append(t) #time
            if grow: #growing, need newest angle, if shrinking, new angle = angle[-1]
                self.angle.append(th) #angle
            self.seg_dist.append(dist(self.seg[-2],self.seg[-1])) #seg length
            if resolve in ['cross', 'cross_bdl', 'top', 'bottom', 'left', 'right','zipper', 'reuse_bdry','catas','freed_br','follow_br', \
                           'grow_to_pause', 'shrink_to_pause']:
                self.traj.append(self.traj[-1])
            elif resolve in ['deflect','follow_bdl']:
                assert traj_id != None
                self.traj.append(traj_id)
        else:
            assert double_check
            self.update_t.append(t) #time
    def rescue_seg(self, t):
        '''
        Updates MT vertices after a rescue.

        Parameters
        ----------
        t : Float
            Time of rescue.

        Returns
        -------
        None.

        '''
        #erases shrunken segments and adds new tip from rescue
        t_i = self.update_t[-1] #initial time
        cumsum = np.append([0],np.cumsum(self.seg_dist)) #cumulative sums
        d = (t- t_i)*v_s #distance traversed
        left_over = cumsum[-1] - d
        j = 0
        while left_over > cumsum[j]: #find index within cumsum
            j+=1
            if j > len(cumsum)-1:
                break
        j -= 1
        delta = left_over - cumsum[j] #component sticking out from segment
        self.seg[:] = self.seg[:j+1] #untouched points in shrinkage
        self.traj[:] = self.traj[:j+1]
        self.angle[:] = self.angle[:j+1] #need to delete all shrunken parts
        self.seg_dist[:] = self.seg_dist[:j]
        th = self.angle[j]
        x, y = self.seg[j][0], self.seg[j][1] #find new pt
        self.seg.append([x + cos(th)*delta, y + sin(th)*delta]) #where it has grown to
        self.traj.append(self.traj[-1])
        self.angle.append(th)
        self.update_t.append(t)
        segment_dist = dist(self.seg[-1],self.seg[-2])
        self.seg_dist.append(segment_dist)
    def step_back_seg(self, new_pt, t): #idea as rescue_seg but with step back due to zippering
        #erases shrunken segments and adds new tip from rescue
        assert self.hit_bdry
        cumsum = np.append([0],np.cumsum(self.seg_dist)) #cumulative sums
        d = dist(new_pt, self.seg[-1]) #distance traversed
        left_over = cumsum[-1] - d
        j = 0
        while left_over > cumsum[j]: #find index within cumsum
            j+=1
            if j > len(cumsum)-1:
                break
        j -= 1
        delta = left_over - cumsum[j] #component sticking out from segment
        self.seg[:] = self.seg[:j+1] #untouched points in shrinkage
        self.traj[:] = self.traj[:j+1]
        self.angle[:] = self.angle[:j+1] #need to delete all shrunken parts
        self.seg_dist[:] = self.seg_dist[:j]
        th = self.angle[j]
        x, y = self.seg[j][0], self.seg[j][1] #find new pt
        self.seg.append([x + cos(th)*delta, y + sin(th)*delta]) #where it has grown to
        self.traj.append(self.traj[-1])
        self.angle.append(th)
        self.update_t.append(t)
        segment_dist = dist(self.seg[-1],self.seg[-2])
        self.seg_dist.append(segment_dist)
    def entrain(self, mt, pt, angle, traj_id):
        '''
        After creation, add geometry of barrier MT to this new one.

        Parameters
        ----------
        mt : mt class
            MT that this MT is following.
        pt : Array of floats
            Pt of creation.
        angle : Foat
            Starting angle.
        traj_id : Int
            Traj ID.

        Returns
        -------
        None.

        '''
        self.prev_t = mt.update_t[-1] #same as update time before bdry
        self.from_bdry = True
        self.prev_mt = mt.number
        self.update_t[-1] = mt.update_t[-1] #give update time
        self.angle.append(angle) #update angle
        self.seg.append(pt)
        self.traj.append(traj_id)
        self.region = which_region(pt)
    def entrain_no_mt(self, t, pt, angle, traj_id):
        '''
        After creation, add geometry of barrier MT to this new one.
        PRALLEL NUCLEATION

        Parameters
        ----------
        pt : Array of floats
            Pt of creation.
        angle : Foat
            Starting angle.
        traj_id : Int
            Traj ID.

        Returns
        -------
        None.

        '''
        self.update_t[-1] = t #give update time
        self.angle.append(angle) #update angle
        self.seg.append(pt)
        self.traj.append(traj_id)
        self.region = which_region(pt)
    def next_path(self, phi0, del_l2, global_p, l_end=6, ds=0.01):
        '''
        Generate path from anchoring & EL.

        Parameters
        ----------
        phi0 : Float
            Initial angle.
        del_l2 : Float
            v_g/k_on parameter.
        global_p: global_prop class
            Used for tracking number of MTs
        l_end : Float, optional
            End length of the MT to be simulated. The default is 0.1.
        ds : Float, optional
            Discretization of the trajectory. The default is 0.01.

        Returns
        -------
        New traj angle.

        '''
        n = self.path_idx
        if n == len(self.path_dist): #if progressed beyond current info, generate more
            if deflect_on: #need to actually defect
                if abs(phi0 - pi/2)<vtol or abs(phi0 - 3*pi/2)<vtol: #initial angle already vertical
                    self.path_angle.append(phi0) #angle same as initial
                    self.path_dist.append(float('inf')) #tip length is treated as infinite before deflection
                else:
                    if len(global_p.grow_mt)+len(global_p.shrink) > 700:
                        l_end = l_end/2 #if many mts, no need to generate a large length
                    # if self.number == 2:
                    #     assert False
                    #     res = gen_path_secant(phi0, self.tip_lf,l_end, del_l2*4, ds)
                    # else:
                    res = gen_path_secant(phi0, self.tip_lf,l_end, del_l2, ds)
                    if self.tip_lf != 0: #keep track of additional anchoring sims
                        global_p.more_anchoring +=1
                    # assert vtol < 2*pi
                    for i in range(len(res[0])):
                        self.path_dist.append(res[1][i])
                        self.path_angle.append(res[0][i]) #careful, res[0] is numpy array
                        assert res[0][i] <= 2*pi
                        if abs(self.path_angle[-1] - pi/2)<vtol or abs(self.path_angle[-1] - 3*pi/2)<vtol:
                            self.path_dist[-1] = float('inf') #nearly vert, no deflection
                            break
                    self.tip_lf = res[2]
            else: #never deflects
                self.path_angle.append(phi0) #angle same as initial
                self.path_dist.append(float('inf')) #tip length is treated as infinite before deflection
        assert n < len(self.path_dist) and n < len(self.path_angle)
        self.tip_l = self.path_dist[n]
        next_angle = self.path_angle[n]
        self.path_idx += 1 #progress to next path
        return(next_angle)
    def erase_path(self): #freed but bundle doesn't change; no new deflections
        self.path_idx  = 0
        self.path_angle = [self.angle[-1]]
        self.path_dist = [float('inf')]
        self.tip_l = self.path_dist[0]
        self.path_idx += 1
    def carry_path(self, mt2):
        '''
        Carry-over path info from prev mt.

        Parameters
        ----------
        mt2 : mt class
            Previous MT that connect to this one.

        Returns
        -------
        None.

        '''
        self.tip_l = mt2.tip_l
        self.path_idx = mt2.path_idx #index of next path to choose
        # n = self.path_idx
        self.path_dist = mt2.path_dist
        self.path_angle = mt2.path_angle
        self.tip_lf = mt2.tip_lf #final tip length from generation, may use for generating more
    def checkd(self):
        '''
        Verify that the segments and recorded lengths are consistent.

        Returns
        -------
        None.

        '''
        tf = True
        for i in range(len(self.seg_dist)):
            if dist(self.seg[i],self.seg[i+1]) !=self.seg_dist[i]:
                tf = False
        return(tf)
    def carry_len(self, mt_prev): #carry over info on mt length
        if self.free:
            mt_prev.total_len += np.sum(mt_prev.seg_dist)
            self.total_len = mt_prev.total_len
            self.init_angle = mt_prev.init_angle
    def check_if_stuck(self,pt, other_region = False): #no bdl case: if forking, check whether it's stuck
        assert no_bdl_id
        pt_prev = None
        if other_region:
            pt_prev = self.seg[0]
        else:
            pt_prev = self.seg[-1]
        distance = dist(pt_prev,pt) #only checks if step back distance is close to previous, might need more sophisticated method
        stuck = False
        # if pt in [[3.4151835770399757, 1.2105502560939239],[3.415183577039977, 1.2105502560939234]]:
        #     print(distance,self.seg[-1])
        if distance < 1e-10: #too close
            stuck = True
        return(stuck)
    def check_pt_topology(self, pt): #check if the step back pt needs to be re-identified according to the topology
        new_pt = pt
        if pt[0] > xdomain[1] or pt[0] < xdomain[0] or pt[1] > ydomain[1] or pt[1] < ydomain[0]:
            bdry_pt = self.seg[0] #point along the bdry the mt comes from
            if bdry_pt[0] == xdomain[0]:
                new_pt = [pt[0]+xdomain[0],pt[1]]
            elif bdry_pt[0] == xdomain[1]:
                new_pt = [pt[0]-xdomain[1],pt[1]]
            elif bdry_pt[1] == ydomain[0]:
                new_pt = [pt[0],pt[1]+ydomain[1]]
            elif bdry_pt[1] == ydomain[1]:
                new_pt = [pt[0],pt[1]-ydomain[1]]
        return(new_pt)

class compare_return:
    '''
    Class for return values of compare function
    '''
    def __init__(self):
        self.policy = None #string: no_collision, 1hit2, 2hit1, 1disap, 2disap, hit_bdry
        self.dist = None #float for collision distance
        self.point = None #point of collision
        self.idx  =None #segment index of collided MT


def dist(p1,p2,square=False):
    '''
    Distance between 2 pts.

    Parameters
    ----------
    p1 : List of floats
        1st pt.
    p2 : List of floats
        2nd pt.
    square : Bool, optional
        Decide whether to return square distance. The default is False.

    Returns
    -------
    Distance between p1,p2 as float
    '''
    dx = p1[0] - p2[0] #calculate distances
    dy = p1[1] - p2[1]
    d = dx*dx + dy*dy
    if square is False:
        return(sqrt(d))
    else:
        return(d)

def tread_dist(d_tr, old_dist, l): #find tread position and vertex
    tr_pos = d_tr #tread pos relative to last vertex
    seg_dist = 0 #total dist of segs considered so far
    seg_start = 0 #last vertex
    for i in range(l-1):
        seg_dist += old_dist[i]
        if d_tr <= seg_dist:
            break
        else:
            seg_start += 1
            tr_pos -= old_dist[i]
    return(tr_pos, seg_start)

def compare(mt1,mt2,t,region_list):
    '''
    Fint earliest collision event between 2 MTs.

    Parameters
    ----------
    mt1 : mt class
        1st MT.
    mt2 : mt class
        2nd MT.
    t : Float
        Current time.
    region_list : List of region_traj
        Trajs for recorded intersections.

    Returns
    -------
    compare_return class

    '''
    output = compare_return() #initiate output
    tol = 1e-13 #numerical tolerance for intersection distance, otherwise no intersection
    r_idx = mt1.region
    assert mt1.region == mt2.region
    if (mt1.hit_bdry is True and mt2.hit_bdry is True) or \
       (mt1.exist is False or mt2.exist is False) or (mt1.hit_bdry is True and mt2.grow is False) or\
       (mt2.hit_bdry is True and mt1.grow is False):
        output.dist = None
        output.policy = 'no_collision'
        output.point = None
        return(output)
    elif (mt1.grow is False and mt2.grow is False): #IF BOTH SHRINKING
        output.policy='no_collision'
    elif (mt1.grow and mt2.grow): #FIRST ONLY LOOKING AT BOTH GROWING
        seg1, seg2 = mt1.seg, mt2.seg #assign segment points
        traj1, traj2 = mt1.traj, mt2.traj
        th1,th2 = mt1.angle, mt2.angle
        old_dist1, old_dist2 = mt1.seg_dist, mt2.seg_dist
        l1, l2 = len(seg1), len(seg2) #for indexing
        p1_prev = seg1[-1] #last updated point and time
        p2_prev = seg2[-1] #last updated point and time
        t_prev1, t_prev2 = mt1.update_t[-1], mt2.update_t[-1]
        assert(len(old_dist1) == len(seg1)-1 and len(old_dist2) == len(seg2)-1)

        col_dist1t2 = [] #collision distances from 1 to 2
        col_dist2t1 = [] #coliision distances from 2 to 1

        point_1t2 = [] #store their respective collision locations
        point_2t1 = []

        seg1_idx = [] #store indices of collision seg_dist
        seg2_idx = []
        #We now check if mt1 collides w/ any segments of mt2
        for i in range(l2):
            p2 = seg2[i] #point traj to be collided with
            col_result = inter2(p1_prev,p2,th1[l1-1],th2[i], traj1[-1], traj2[i], r_idx, region_list)
            if col_result[0] is True:
                d1, d2 = col_result[2][0], col_result[2][1] #distance to collision from mt1 and mt2
                pt = col_result[1] #point of collision
                if i==l2-1: #checking collision w/ the two growing ends
                    d_g1 = d1 - (t-t_prev1) #distance at each mt collision
                    d_g2 = d2 - (t-t_prev2)
                    d_g = max(d_g1,d_g2) #distance grown by mt's is always max
                    if d_g >= tol: #cannot be an intersection that just happened
                        if d_g==d_g1: #store whichever collides with which
                            store = True
                            if mt2.tread:# and i==seg_start2 and d2<tr_pos2:
                                d_tr2 = (d1+t_prev1-mt2.tread_t)*mt2.vt #amount treaded
                                tread_res = tread_dist(d_tr2, old_dist2, l2)
                                tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                                if (seg_start2 > i) or (seg_start2 == i and d2 < tr_pos2):
                                    store = False #treadmilled past the pt
                            if store:
                                col_dist1t2.append(d_g)
                                point_1t2.append(pt)
                                seg2_idx.append(i)
                        else:
                            store = True
                            if mt1.tread: #and (l1-1)==seg_start1 and d1<tr_pos1:
                                d_tr1 = (d2+t_prev2-mt1.tread_t)*mt1.vt #amount treaded
                                tread_res = tread_dist(d_tr1, old_dist1, l1)
                                tr_pos1, seg_start1 = tread_res[0], tread_res[1]
                                if (seg_start1 > (l1-1)) or (seg_start1 == (l1-1) and d1 < tr_pos1):
                                    store = False #treadmilled past the pt
                            if store:
                                col_dist2t1.append(d_g)
                                point_2t1.append(pt)
                                seg1_idx.append(l1-1)
                else:#check intersection of mt1 head w/ previous mt2 segments
                    if d2<= old_dist2[i]: #must be less than segment length for collision
                        d_g = d1-(t-t_prev1) #total distance grown since t
                        if d_g >= tol:
                            store = True
                            if mt2.tread:# and i==seg_start2 and d2<tr_pos2:
                                d_tr2 = (d1+t_prev1-mt2.tread_t)*mt2.vt #amount treaded
                                tread_res = tread_dist(d_tr2, old_dist2, l2)
                                tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                                if (seg_start2 > i) or (seg_start2 == i and d2 < tr_pos2):
                                    store = False #treadmilled past the pt
                            if store:
                                col_dist1t2.append(d_g) #store collision distance
                                point_1t2.append(pt)
                                seg2_idx.append(i)
        #check if m2 collides w/ any segments of mt1
        for j in range(l1):
            if j!=l1-1: #ignore collision w/ dynamic ends -already found above
                p1 = seg1[j] #point traj to be collided with
                col_result = inter2(p2_prev,p1,th2[l2-1],th1[j], traj2[-1], traj1[j], r_idx, region_list)
                pt = col_result[1]
                if col_result[0] is True:
                    d2, d1 = col_result[2][0], col_result[2][1] #distance to collision from mt2 and mt1
                    if d1<= old_dist1[j]: #must be less than segment length for collision
                        d_g = d2-(t-t_prev2) #total distance grown since t
                        if d_g >= tol:
                            store = True
                            if mt1.tread:# and j==seg_start1 and d1<tr_pos1:
                                d_tr1 = (d2+t_prev2-mt1.tread_t)*mt1.vt #amount treaded
                                tread_res = tread_dist(d_tr1, old_dist1, l1)
                                tr_pos1, seg_start1 = tread_res[0], tread_res[1]
                                if (seg_start1 > j) or (seg_start1 == j and d1 < tr_pos1):
                                    store = False #treadmilled past the pt
                            if store:
                                col_dist2t1.append(d_g) #store collision distance
                                point_2t1.append(pt)
                                seg1_idx.append(j)
        which_min, i_1t2, i_2t1 = None, None, None #declare
        if len(col_dist1t2) ==0 and len(col_dist2t1) ==0:
            output.policy = 'no_collision'
            return(output)
        elif len(col_dist1t2) ==0: #if no collisions
            which_min = 1
            i_2t1 = np.argmin(col_dist2t1)
        elif len(col_dist2t1) ==0:
            which_min = 0
            i_1t2 = np.argmin(col_dist1t2)
        else:
            i_1t2 = np.argmin(col_dist1t2) #find min distances
            min_1t2 = col_dist1t2[i_1t2]
            i_2t1 = np.argmin(col_dist2t1)
            min_2t1 = col_dist2t1[i_2t1]

            glob_min = [min_1t2,min_2t1] #array for total min
            which_min = np.argmin(glob_min) #find which occurs
        if which_min==0: #if mt1 hits mt2 first
            if col_dist1t2[i_1t2]<=0: #TODO this should never happen, what's going on here?
                output.policy = 'no_collision'
            else:
                output.policy = '1hit2'
                output.point = point_1t2[i_1t2]
                output.dist = col_dist1t2[i_1t2]
                output.idx = seg2_idx[i_1t2]
        else:
            if col_dist2t1[i_2t1]<=0: #TODO same here
                output.policy = 'no_collision'
            else:
                output.policy = '2hit1'
                output.point = point_2t1[i_2t1]
                output.dist = col_dist2t1[i_2t1]
                output.idx = seg1_idx[i_2t1]
    elif (not mt1.hit_bdry and not mt2.hit_bdry) is True: #IF ONE END IS SHRINKING
        assert mt1.grow is False or mt2.grow is False
        which = None #which tip hits
        if mt1.grow is False: #figure which is shrinking
            mts, mtg = mt1, mt2 #assign
            which = 2
        else:
            mts, mtg = mt2, mt1
            which = 1
        assert len(mts.seg) >= 2
        seg1, seg2 = mtg.seg, mts.seg #assign segment points
        traj1, traj2 = mtg.traj, mts.traj
        th1,th2 = mtg.angle, mts.angle
        old_dist1, old_dist2 = mtg.seg_dist, mts.seg_dist
        l1, l2 = len(seg1), len(seg2) #for indexing
        t_prev1, p1_prev = mtg.update_t[-1], seg1[l1-1] #last updated point and time
        t_prev2, p2_prev = mts.update_t[-1], seg2[l2-1] #last updated point and time
        p1_prev = seg1[-1] #last updated point and time
        p2_prev = seg2[-1] #last updated point and time
        assert(len(old_dist1) == len(seg1)-1)
        assert(len(old_dist2) == len(seg2)-1)

        mt2_l = np.sum(old_dist2)#total length of shrinking mt, cannot got lower than this
        col_dist1t2 = [mt2_l/v_s - (t-t_prev2)] #collision distances from 1 to 2, growing can only collide w/ shrinking
        point_1t2 = [[0,0]] #TODO why is this here?

        seg2_idx = [0]
        #We now check if mt1 collides w/ any segments of mt2
        for i in range(l2-1): #only up to second last pt matters when shriking
            if i==l2-2: #checking collision w/ the two dynamic ends
                p2 = seg2[i] #point traj to be collided with
                col_result = inter2(p1_prev,p2,th1[l1-1],th2[i], traj1[-1], traj2[i], r_idx, region_list)
                if col_result[0] is True:
                    d1, d2 = col_result[2][0], col_result[2][1] #distance to collision from mt1 and mt2
                    if d2<= old_dist2[i]: #has to collide on segment
                        d_g = d1 - (t-t_prev1) #distance of mt1 collision, also time taken to grow this
                        d_s = v_s*(t-t_prev2+d_g) #distance shrank
                        d_segf = old_dist2[i] - d_s #total distance left on the segment
                        assert d2 != d_segf
                        if d2 < d_segf:
                            if d_g >= tol:
                                store = True
                                if mts.tread:# and i==seg_start2 and d2<tr_pos2:
                                    d_tr2 = (d1+t_prev1-mts.tread_t)*mts.vt #amount treaded
                                    tread_res = tread_dist(d_tr2, old_dist2, l2)
                                    tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                                    if (seg_start2 > i) or (seg_start2 == i and d2 < tr_pos2):
                                        store = False #treadmilled past the pt
                                if store:
                                    col_dist1t2.append(d_g)
                                    point_1t2.append(col_result[1])
                                    seg2_idx.append(i)
            else:#check intersection of mt1 head w/ previous mt2 segments
                p2 = seg2[i] #point traj to be collided with
                col_result = inter2(p1_prev,p2,th1[l1-1],th2[i], traj1[-1], traj2[i], r_idx, region_list)
                if col_result[0] is True:
                    d1, d2 = col_result[2][0], col_result[2][1] #distance to collision from mt1 and mt2
                    if d2<= old_dist2[i]: #has to collide on segment
                        d_g = d1 - (t-t_prev1) #distance/time taken to grow starting at t
                        d_s = v_s*(t-t_prev2+d_g) #distance shrank of mt2
                        d_segf = mt2_l - d_s #length of mt2 left
                        for k in range(i): #total length of mt2 at intersection point, add prev segs
                            d2 += old_dist2[k]
                        if d_g >= tol:
                            assert d2 != d_segf
                        if d2< d_segf: #if shrank to less than intersection distance
                            if d_g >= tol:
                                store = True
                                if mts.tread:# and i==seg_start2 and d2<tr_pos2:
                                     d_tr2 = (d1+t_prev1-mts.tread_t)*mts.vt #amount treaded
                                     tread_res = tread_dist(d_tr2, old_dist2, l2)
                                     tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                                     if (seg_start2 > i) or (seg_start2 == i and col_result[2][1] < tr_pos2): #d2 was edited, use original value
                                         store = False #treadmilled past the pt
                                if store:
                                    col_dist1t2.append(d_g)
                                    point_1t2.append(col_result[1])
                                    seg2_idx.append(i)
        i_1t2 = None #declare
        if len(col_dist1t2)==1: #if not additional points are added, mt shrinks away before collision
            output.dist = col_dist1t2[0]
            if which == 1:
                output.policy = 'no_collision'#'2disap'
            else:
                output.policy = 'no_collision'#'1disap'
        else: #collision occurs
            i_1t2 =  np.argmin(col_dist1t2)
            min_1t2 = col_dist1t2[i_1t2]
            output.point = point_1t2[i_1t2]
            output.dist = col_dist1t2[i_1t2]
            output.idx = seg2_idx[i_1t2]
            if which == 2:
                output.policy = '2hit1'
            else:
                output.policy = '1hit2'
    elif (mt1.hit_bdry or mt2.hit_bdry) is True: #IF ONE IS GROWING AND ONE IS ON THE BORDER
        assert (mt1.grow or mt2.grow) is True
        which = None
        if mt1.hit_bdry is True: #figure which is growing
            mtb, mtg = mt1, mt2 #assign
            which = 2
        else:
            which = 1
            mtb, mtg = mt2, mt1
        assert len(mtb.seg) >= 2
        seg1, seg2 = mtg.seg, mtb.seg #assign segment points
        traj1, traj2 = mtg.traj, mtb.traj
        th1,th2 = mtg.angle, mtb.angle
        old_dist1, old_dist2 = mtg.seg_dist, mtb.seg_dist
        l1, l2 = len(seg1), len(seg2) #for indexing
        t_prev1, p1_prev = mtg.update_t[-1], seg1[l1-1] #last updated point and time
        t_prev2, p2_prev = mtb.update_t[-1], seg2[l2-1] #last updated point and time
        p1_prev = seg1[-1] #last updated point and time
        p2_prev = seg2[-1] #last updated point and time
        assert(len(old_dist1) == len(seg1)-1 and len(old_dist2) == len(seg2)-1)
        # assert mt1.checkd() and mt2.checkd()
        col_dist1t2 = [] #collision distances from 1 to 2
        point_1t2 = [] #store their respective collision locations
        neg_dist = 0

        seg2_idx=[]
        #We now check if mt1 collides w/ any segments of mt2
        for i in range(l2-1):
            p2 = seg2[i] #point traj to be collided with
            col_result = inter2(p1_prev,p2,th1[l1-1],th2[i], traj1[-1], traj2[i], r_idx, region_list)
            if col_result[0] is True:
                d1, d2 = col_result[2][0], col_result[2][1] #distance to collision from mt1 and mt2
                pt = col_result[1]
                if d2<= old_dist2[i]: #must be less than segment length for collision
                    d_g = d1-(t-t_prev1) #total distance grown since t
                    if d_g<= tol:
                        neg_dist += 1
                    else:
                        store = True
                        if mtb.tread:# and i==seg_start2 and d2<tr_pos2:
                            d_tr2 = (d1+t_prev1-mtb.tread_t)*mtb.vt #amount treaded
                            tread_res = tread_dist(d_tr2, old_dist2, l2)
                            tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                            if (seg_start2 > i) or (seg_start2 == i and d2 < tr_pos2):
                                store = False #treadmilled past the pt
                            # store = False #treadmilled past the pt
                        if store:
                            col_dist1t2.append(d_g) #store collision distance
                            point_1t2.append(pt)
                            seg2_idx.append(i)
        if len(col_dist1t2)==0 or neg_dist>0:
            output.policy = 'no_collision'
        else:
            i_1t2 = np.argmin(col_dist1t2)
            min_1t2 = col_dist1t2[i_1t2]
            output.point = point_1t2[i_1t2]
            output.dist = col_dist1t2[i_1t2]
            output.idx = seg2_idx[i_1t2]
            if which == 2:
                output.policy = '2hit1'
            else:
                output.policy = '1hit2'
    if output.policy != 'no_collision':
        assert output.dist > tol
        x = x_interval[mt1.region] #discard events outside of region
        y = y_interval[mt2.region]
        xi, xf = x[0], x[1]
        yi, yf = y[0], y[1]
        if output.point[0] >= xf or output.point[0] <= xi or output.point[1] >= yf or output.point[1] <= yi:
            output.policy = 'no_collision'
    return(output)


def which_region(pt):
    '''
    Which grid the point lies in.

    Parameters
    ----------
    pt : List of floats
        Pt.

    Returns
    -------
    Grid index.

    '''
    x, y = pt[0], pt[1]
    r_idx = np.floor(x/dx) + np.floor(y/dy)*grid_w
    r_idx = int(r_idx)
    return(r_idx)

mt_none = mt(None) #dummy mt for None placeholder in event(mt1,mt2=None,...)

def arc_pos(pt, bdl_pol, seg, seg_dist):
    '''
    Find arc length given pt on MT [Psi^(-1)]
    Can do because mapping is invtertible (relies on monotonicity).
    Assumes point is beyond the first vertex, in the plus dir of the MT.
    Otherwise, simply calculates the distance from the first vertex point with the opposide sign.

    Parameters
    ----------
    pt : List of floats
        Pt.
    bdl_pol : Int
        Bdl polarity.
    seg : List of list of floats
        List of vertices of MT.
    seg_dist : List of floats
        Already calculated segment dists.

    Returns
    -------
    Float, position on premimage.

    '''
    x = pt[0] #x coord
    pos = 0 #total length
    n = len(seg)
    if bdl_pol > 0: # + direction
        if seg[0][0] <= x:
            if n == 1: #kind of roundabout, special case for 1 element
                pos = dist(seg[0],pt)
            else:
                for i in range(1,n):
                    if seg[i][0] < x: #monotonicity
                        pos += seg_dist[i-1] #add whole seg
                        if i == n-1: #case when pt is beyond existing vertices
                            pos += dist(seg[n-1],pt)
                    else: #add intermediate length
                        if seg[i][0] != x: #only if pt isn't exactly on seg
                            pos += dist(seg[i-1],pt)
                        else: #if exactly on seg, reuse dist
                            pos += seg_dist[i-1]
                        break #end calc in this case
        else:
            assert no_bdl_id
            pos = -dist(seg[0],pt)
    else: # - direction
        if seg[0][0] >= x:
            if n == 1: #kind of roundabout, special case for 1 element
                pos = dist(seg[0],pt)
            else:
                for i in range(1,n):
                    if seg[i][0] > x: #add whole seg
                        pos += seg_dist[i-1]
                        if i == n-1: #case when pt is beyond existing vertices
                            pos += dist(seg[n-1],pt)
                    else: #add intermediate length
                        if seg[i][0] != x: #only if pt isn't exactly on seg
                            pos += dist(seg[i-1],pt)
                        else: #if exactly on seg, reuse dist
                            pos += seg_dist[i-1]
                        break #end calc in this case
            pos *= -1
        else:
            assert no_bdl_id
            pos = dist(seg[0],pt)
    return(pos)

def arc_pt(pos, seg, angle, seg_dist):
    '''
    Find pt on bdl given its arc position, [Psi].

    Parameters
    ----------
    pos : Float
        Position of bdl.
    seg : List of list of floats
        List of vertices of MT.
    angle : List of floats
        List of angles of MT.
    seg_dist : List of floats
        Already calculated segment dists.

    Returns
    -------
    Point, as list of floats.

    '''
    current = 0 #current distance
    total = 0 #keep track of running sum
    j = 0 #to keep track of index
    l = len(seg_dist)
    pos = abs(pos) #taking abs. distance here
    for i in range(l): #find seg it's on
        # print(current, pos, seg_dist)
        if current <= pos:
            total = current
            current += seg_dist[i]
            j+=1
        else:
            break
    if current <= pos and l>=1: #if position still greater than existant seg data
        total += seg_dist[l-1]
        j+=1
    j -= 1 #seg it's on
    ds = pos - total #difference left  over on the seg
    pt = seg[j] #seg pt
    th = angle[j] #direction of seg
    # print(th)
    pt2 = [pt[0] + ds*cos(th), pt[1] + ds*sin(th)] #new point
    return(pt2)

def deflect_angle(mt1,frac=10):
    '''
    Placeholder for deflection angle

    Parameters
    ----------
    mt1 : TYPE
        DESCRIPTION.
    frac : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    '''
    dth = 0 #change in angle
    if mt1.angle[-1] < pi:
        dth = (pi/2 - mt1.angle[-1])/frac
    else:
        dth = (3*pi/2 - mt1.angle[-1])/frac
    th_new = mt1.angle[-1] + dth
    return(th_new)

def branch_geo(ref_angle, branch_angle, branch_in): #find the branch geometry
    NS = None #north or south
    EW = None #east or west
    if ref_angle <= pi/2:
        if branch_angle < ref_angle + pi/2 or branch_angle > ref_angle+3*pi/2:
            NS = 'north'
            if branch_angle < ref_angle + pi/2 and branch_angle > ref_angle:
                EW = 'east'
            else:
                EW = 'west'
        else:
            NS = 'south'
            if branch_angle < ref_angle + pi and branch_angle > ref_angle + pi/2:
                EW = 'east'
            else:
                EW = 'west'
    elif ref_angle > pi/2 and ref_angle <= pi:
        if branch_angle > ref_angle - pi/2 and branch_angle < ref_angle + pi/2:
            NS = 'north'
            if branch_angle < ref_angle:
                EW = 'west'
            else:
                EW = 'east'
        else:
            NS = 'south'
            if branch_angle < ref_angle + pi and branch_angle > ref_angle + pi/2:
                EW = 'east'
            else:
                EW = 'west'
    elif ref_angle >pi and ref_angle <= 3*pi/2:
        if branch_angle > ref_angle-pi/2 and branch_angle < ref_angle + pi/2:
            NS = 'north'
            if branch_angle  < ref_angle:
                EW = 'west'
            else:
                EW = 'east'
        else:
            NS = 'south'
            if branch_angle > ref_angle - pi and branch_angle < ref_angle - pi/2:
                EW = 'west'
            else:
                EW = 'east'
    elif ref_angle > 3*pi/2 and ref_angle <= 2*pi:
        if branch_angle < ref_angle-pi/2 and branch_angle > ref_angle - 3*pi/2:
            NS = 'south'
            if branch_angle > ref_angle - pi:
                EW = 'west'
            else:
                EW = 'east'
        else:
            NS = 'north'
            if branch_angle < ref_angle -3*pi/2 or branch_angle > ref_angle:
                EW = 'east'
            else:
                EW = 'west'
    assert NS != None and EW != None
    if not branch_in: #if the branch is going outward, sidedness is reversed but not polarity
        if EW == 'west':
            EW = 'east'
        else:
            EW = 'west'
    return(NS,EW)

class branch:
    '''
    Branch info
    ref_bdl: the bdl that the branch comes out of
    branch_bdl: the bdl of the branch
    '''
    def __init__(self,ref_bdl, br_bdl, mt, mt_br, branch_list, branch_in, mt_list, region_list, bdl_list, t, natural = False, origin=True, special=False, branching = False):
        self.nucleated = False #whether nucleated. Nucleated branches can have deep angles!
        self.shallow = True #if nucleated, whether angle is shallow
        if branching: #in this case, will edit sidedness and level outside of this method
            self.nucleated = True
            self.number = len(branch_list) #creation of new branch in list
            self.ref_bdln = ref_bdl.number #origin bdl of branch
            self.branch_bdln = br_bdl.number #bdl branch is entrained to
            #get mt index for sidedness
            # if not branch_in: #exiting ref bdl, its sidedness is the same as original mt
                # mt_i = ref_bdl.mts.index(mt.number)
                # self.level = ref_bdl.mt_sides[mt_i] #add sidedness of mts to bdl class
                # self.side = compass[1] 
            # if branch_in: #sidedness is the max/min of intersecting mts' sidedness
            #         pt = mt.seg[-1] #pt of intersection
            #         overlap = ref_bdl.mt_overlap(mt_list, pt, t, mt_none=[mt_br.number])
            #         mt_sided  = [d for idx, d in enumerate(ref_bdl.mt_sides) if ref_bdl.mts[idx] in overlap ] #get list of sidedness
            #         if compass[1] == 'east':
            #             self.level = max(mt_sided) + 1
            #             self.side = 'east'
            #         else:
            #             self.level = min(mt_sided) - 1
            #             self.side = 'west'
            self.side = None
            self.level = None
            self.mts = [] #mts on the branch
            self.angle = None
            # self.traj = mt_br.traj[-1]
            self.branch_in = branch_in#whether going in to ref bdl or not
            self.branch_pol = None
            self.natural = natural #whether branch is natural
            self.twin = self.number+1 #corresponding branch on other bdl
            if not origin:
                self.twin = self.number-1
        elif not special: #not a pseudo branch
            region = region_list[mt.region]
            ref_angle = region.angle[mt.traj[-1]] #reference angle as defined by the bdl
            branch_angle = mt_br.angle[-1]
            self.number = len(branch_list) #creation of new branch in list
            self.ref_bdln = ref_bdl.number #origin bdl of branch
            self.branch_bdln = br_bdl.number #bdl branch is entrained to
            compass = branch_geo(ref_angle, branch_angle, branch_in) #sidedness info
            #get mt index for sidedness
            if not branch_in: #exiting ref bdl, its sidedness is the same as original mt
                mt_i = ref_bdl.mts.index(mt.number)
                self.level = ref_bdl.mt_sides[mt_i] #add sidedness of mts to bdl class
                self.side = compass[1] 
            else: #sidedness is the max/min of intersecting mts' sidedness
                if natural:# and vtol == 2*pi: #side info not applicable to pseudo branches
                    self.side = None
                    self.level = None
                else:
                    pt = mt.seg[-1] #pt of intersection
                    overlap = ref_bdl.mt_overlap(mt_list, pt, t, mt_none=[mt_br.number])
                    mt_sided  = [d for idx, d in enumerate(ref_bdl.mt_sides) if ref_bdl.mts[idx] in overlap ] #get list of sidedness
                    # if len(mt_sided) == 0:
                    #     #this can occur due to rare floating point problems
                    #     #to fix, we ignore such case and let it continue
                    #     warnings.warn('Pathological case found for MT side calculation, current time is {}'.format(t))
                    #     mt_sided = [0]
                    if compass[1] == 'east':
                        self.level = max(mt_sided) + 1
                        self.side = 'east'
                    else:
                        self.level = min(mt_sided) - 1
                        self.side = 'west'
            self.mts = [mt_br.number] #mts on the branch
            self.angle = branch_angle
            self.traj = mt_br.traj[-1]
            self.branch_in = branch_in#whether going in to ref bdl or not
            self.branch_pol = compass[0]
            self.natural = natural #whether branch is natural
            self.twin = self.number+1 #corresponding branch on other bdl
            if not origin:
                self.twin = self.number-1
        if special: #pseudo branch representing a stationary negative end
            self.number = 0
            self.ref_bdln = None
            self.branch_bdln = None
            self.level = None
            self.side = None
            self.mts = None
            self.angle = None
            self.traj = None
            self.branch_in = False#whether going in to ref bdl or not
            self.branch_pol = 'south'
            self.natural = True #whether branch is natural
            self.twin = 0
    def add_mt_mid(self, mt, mt_br, ref_bdl, branch_in = False, overlap = None): #branch mt added from collision or following branch
        if not branch_in: #branch goes out (in the original context of mt, mt_br)
            mt_i = ref_bdl.mts.index(mt.number)
            mt_side = ref_bdl.mt_sides[mt_i]
            self.mts.append(mt_br.number)
            overlap_check = False #mt 'feels' overlapping mts
            if (self.side == 'east' and mt_side <= self.level) or (self.side == 'west' and mt_side >= self.level): #if branch level gets promoted/demoted
                self.level = mt_side #TODO equality above makes physical sense?
                overlap_check = True
            return(overlap_check)
        else: #branch goes in on the twin
            self.mts.append(mt_br.number)
            mt_side = None
            mt_sides  = [d for idx, d in enumerate(ref_bdl.mt_sides) if ref_bdl.mts[idx] in overlap]
            assert len(mt_sides) > 0
            #check if branch level changes
            if self.side =='east':
                mt_side = max(mt_sides)+1
                if mt_side < self.level:
                    self.level = mt_side 
            else:
                mt_side = min(mt_sides)-1
                if mt_side > self.level:
                    self.level = mt_side 
            ref_bdl.mt_sides.append(mt_side)
    def add_mt_br(self, mt, mt_br, ref_bdl, bdl_br, branch_list, mt_list, branch_in = False, root = False, fake = False): #add mt due to collision w/ branch, not bdl
        #fake means whether the branch is natural, in which it's not really a "real" branch but simply an extension.
        #in such case, the sidedness is endowed by the previous mt bundle
        if branch_in:
            if not fake:
                branch2 = branch_list[self.twin]
                mtbr_i = bdl_br.mts.index(mt_br.number) #find sidedness of branch mt on its bdl
                mtbr_side = bdl_br.mt_sides[mtbr_i]
                barrier_lvl = [] #mts on ref_bdl to compare branch_level
                mt_lvl = None
                for mt2_n in self.mts: #check sidedness of other mts in this branch
                    mt2_i = bdl_br.mts.index(mt2_n)
                    mt2_side = bdl_br.mt_sides[mt2_i]
                    mt2 = mt_list[mt2_n]
                    if branch2.side == 'east':
                        if mt2_side < mtbr_side:
                            if mt_list[mt2_n].ext_mt in ref_bdl.mts:
                                lvl_i = ref_bdl.mts.index(mt2.ext_mt)
                                lvl = ref_bdl.mt_sides[lvl_i]
                                barrier_lvl.append(lvl)
                            elif mt_list[mt2_n].prev_mt in ref_bdl.mts:
                                lvl_i = ref_bdl.mts.index(mt2.prev_mt)
                                lvl = ref_bdl.mt_sides[lvl_i]
                                barrier_lvl.append(lvl)
                    else:
                        assert branch2.side == 'west'
                        if mt2_side > mtbr_side:
                            if mt_list[mt2_n].ext_mt in ref_bdl.mts:
                                lvl_i = ref_bdl.mts.index(mt2.ext_mt)
                                lvl = ref_bdl.mt_sides[lvl_i]
                                barrier_lvl.append(lvl)
                            elif mt_list[mt2_n].prev_mt in ref_bdl.mts:
                                lvl_i = ref_bdl.mts.index(mt2.prev_mt)
                                lvl = ref_bdl.mt_sides[lvl_i]
                                barrier_lvl.append(lvl)
                if not root and not self.nucleated: #if adding mt to root end, barrier may not exist
                    #nucleated case special because it's possible to not have mts
                    assert len(barrier_lvl) > 0 #should exist some branch as barrier
                if not self.nucleated:
                    if len(barrier_lvl) == 0: #special rare case, arbitrarily set level to 0
                        mt_lvl = 0
                    if self.side == 'east':
                        mt_lvl = max(barrier_lvl)+1
                        assert mt_lvl >= self.level
                    else:
                        mt_lvl = min(barrier_lvl)-1
                        assert mt_lvl <= self.level
                    self.mts.append(mt_br.number)
                    # ref_bdl.mts.append(mt.number)
                    ref_bdl.mt_sides.append(mt_lvl)
                if self.nucleated:
                    if len(barrier_lvl) == 0: #requires us to consider crossed mts
                        return(True) #we do not append mts or mt_sides in this case
                    else:
                        if self.side == 'east':
                            mt_lvl = max(barrier_lvl)+1
                            assert mt_lvl >= self.level
                        else:
                            mt_lvl = min(barrier_lvl)-1
                            assert mt_lvl <= self.level
                        self.mts.append(mt_br.number)
                        # ref_bdl.mts.append(mt.number)
                        ref_bdl.mt_sides.append(mt_lvl)
                        return(False)
            else:
                self.mts.append(mt_br.number)
                mtbr_i = bdl_br.mts.index(mt_br.number) #find sidedness of branch mt on its bdl
                mtbr_side = bdl_br.mt_sides[mtbr_i]
                mtbr_pol = bdl_br.rel_polarity[mtbr_i]
                if mtbr_pol == bdl_br.pol:
                    ref_bdl.mt_sides.append(mtbr_side)
                else:
                    ref_bdl.mt_sides.append(-mtbr_side)
        else:
            self.mts.append(mt_br.number)
    def remove_mt(self, mt_br, mt_on, ref_bdl, mt_list): #tread causes re-naming of root mt
        assert ref_bdl.number == self.ref_bdln
        self.mts.remove(mt_br.number)
        assert mt_br.number not in self.mts
        idx = ref_bdl.mts.index(mt_on.number)
        mt_side = ref_bdl.mt_sides[idx]
        if mt_side == self.level and len(self.mts) != 0: #TODO compares int with possibly None if root ext disappears
            lvl = [] 
            for mtn in self.mts:
                mt1 = mt_list[mtn]
                if (mt1.ext_mt in ref_bdl.mts):
                    lvl_i = ref_bdl.mts.index(mt1.ext_mt)
                    lvl.append(ref_bdl.mt_sides[lvl_i])
                elif (mt1.prev_mt in ref_bdl.mts):
                    lvl_i = ref_bdl.mts.index(mt1.prev_mt)
                    lvl.append(ref_bdl.mt_sides[lvl_i])
            if self.side == 'east':
                self.level = min(lvl)
            else:
                self.level = max(lvl)
        if len(self.mts) == 0:
            if self.side == 'east':
                self.level = np.inf
            else:
                self.level = -np.inf
    def reset_level(self): #only called after branched nucleation with treadmilling, automatically doesn't have any mts in branch
        assert len(self.mts) == 0
        if self.side == 'east':
            self.level = np.inf
        else:
            self.level = -np.inf

class bundle:
    '''
    Class for bundles
    '''
    def __init__(self, mt, bdl_list, event = None, Policy=None):
        '''
        Initialize bdl, with mt as the parent MT.

        Parameters
        ----------
        mt : mt class
            Parent MT.
        bdl_list : List of bld class
            List of bdls.
        event : Event class, optional
            Event. The default is None.
        Policy : String, optional
            Event policy, not always same as above (1hit2 result). The default is None.

        Returns
        -------
        None.

        '''
        pol = 1 #assign polarity
        if mt.angle[0] > pi/2 and mt.angle[0] < 3*pi/2:
            pol = -1
        self.pol = pol #bundle polarity, same as initial mt but initial mt info may be deleted
        #bdl ID info
        if len(bdl_list) == 0:
            self.number = 0
        else:
            self.number = bdl_list[-1].number + 1    
        self.pseudo_bdry = False
        self.seg = [mt.seg[0]] #original vertex segments of reference mt, 2D
        self.angle = [mt.angle[0]]
        self.seg_dist = [] #for following deflections
        self.traj = [mt.traj[0]]
        self.region = mt.region
        for i in range(1,len(mt.angle)): #if there are any repetitions on the same line, ignore
            special = False #below gets rid of bdry pts which are still needed, need to exempt this case
            if Policy in ['top','bottom', 'left', 'right'] and i == len(mt.angle)-1:
                special = True
            if (mt.angle[i-1] != mt.angle[i]) or special: #it's equal if there's a cross pt, causes double counting w/ deflect events
                self.seg.append(mt.seg[i])
                self.angle.append(mt.angle[i])
                self.traj.append(mt.traj[i])
        for i in range(1,len(self.angle)): #append distances
            self.seg_dist.append(dist(self.seg[i-1],self.seg[i]))
        #info on added mts
        self.mts = [mt.number] #MTs within the bundle (parallel)
        self.rel_polarity = [pol] #direction of MT growth in relation to bundle dir(+1 for same dir) -- defined globally NOT relatively
        self.start_pos  = [0] #position of mt start growth in arclength coords
        #crossover info
        self.cross_bdl = [] #crossover bundle if exists
        self.cross_mts = [] #crossover mts corresponding to bdl if exists
        self.cross_pos = [] #crossover positions if exists
        self.cross_pt = []
        #branch info
        self.mt_sides = [] #sideness
        self.branch_pos = []
        self.branch_pt = []
        self.branchn = []
        #bdry info
        self.ext_bdl = None #extension if exists
        self.prev_bdl = None #previous if exists
        self.ext_pt = None #bdry pts
        self.prev_pt = None
        self.ext_traj = None
        self.prev_traj = None
        self.free_start = True #whether starting end is not from bdry
        if mt.from_bdry:
            self.free_start = False
        self.policy = Policy
        #edit mt properties
        mt.bdl = self.number
        if self.free_start:
            self.mt_sides.append(0)
            self.branch_pos.append(0)
            self.branch_pt.append(self.seg[0])
            self.branchn.append([0])
        if event is None: #bdl created w/ out bdry collision, need to calculate both bdry pts
            0
        elif mt.number == event.mt1_n: #bdl created at bdry, reuse pt
            if mt.angle[0] > pi/2 and mt.angle[0] < 3*pi/2:
                self.prev_pt = copy.deepcopy(event.pt)
            else:
                self.ext_pt = copy.deepcopy(event.pt)
        else:
            assert mt.number is not None
            if mt.angle[0] > pi/2 and mt.angle[0] < 3*pi/2:
                self.ext_pt = copy.deepcopy(mt.seg[0])
            else:
                self.prev_pt = copy.deepcopy(mt.seg[0])
        if mt.pseudo_bdry:
            self.pseudo_bdry = True
            self.mt_sides.append(0)
            self.ext_pt = mt.seg[-1]
            self.prev_pt = mt.seg[0]
            self.ext_traj = mt.traj[0]
            self.prev_traj = mt.traj[0]
            self.seg.append(mt.seg[-1])
            self.angle.append(mt.angle[-1])
            self.seg_dist.append(mt.seg_dist[0])
            up = False
            grid_diff = 0
            if mt.number >= grid_w:
                up = True
                grid_diff = grid_w
            if mt.number in [0, grid_w]:
                self.prev_bdl = mt.number + grid_w - 1
                self.ext_bdl = mt.number + 1
            elif mt.number in [grid_w-1, 2*grid_w-1]:
                self.prev_bdl = mt.number - 1
                self.ext_bdl = mt.number + 1 - grid_w
            else:
                self.ext_bdl = mt.number + 1
                self.prev_bdl = mt.number - 1
        if Policy in ['entrain_spaced', 'entrain_other_region']:
            self.mt_sides.append(0)
            assert len(self.mt_sides) == 1
    def add_seg(self, pt, th, traj_id):
        '''
        Leading mt adds new vertex from deflection.

        Parameters
        ----------
        pt : List of floats
            New vertex.
        th : Float
            New angle.
        traj_id : Int
            ID of new traj.

        Returns
        -------
        None.

        '''
        self.seg.append(pt)
        self.angle.append(th)
        self.seg_dist.append(dist(self.seg[-1],self.seg[-2]))
        self.traj.append(traj_id)
    def get_side(self, mtn):
        if no_bdl_id:
            return(0)
        else:
            idx = self.mts.index(mtn)
            return(self.mt_sides[idx])
    def add_branch_bdl(self, pt, bdl2, mt1, mt_br, angle, inward, branch_list, mt_list, region_list,bdl_list, t, natural=False, origin=False, branching = False):
        '''
        Add new branch info to bdl.

        Parameters
        ----------
        pt : List of floats
            Branch pt.
        bdl : bundle class
            Bundle of interest.
        mt : mt class
            MT that create the branch.
        angle : Float
            MT traj angle.
        inward : Bool
            Whether MT is incoming, for determining geometry.
        branching : Bool
            Whether it's from branch nucleation

        Returns
        -------
        None.

        '''
        bdl1 = bdl_list[self.number]
        assert len(self.branch_pos) == len(self.branch_pt)
        if pt in self.branch_pt: #can exist due to numerical precision BUT the branch should not be occupied
            assert not branching
            pt_i = self.branch_pt.index(pt)
            assert len(self.branchn[pt_i]) <= 2 #TODO double check this, is the mt angle the unique indicator?
            for i in range(len(self.branchn[pt_i])): #prevent double counting of branches due to numerical precision
                br_n = self.branchn[pt_i][i]
                br = branch_list[br_n]
                if br.angle == angle and inward == br.branch_in:
                    assert len(br.mts) == 0 
                    del self.branchn[pt_i][i]
            branch_list.append(branch(bdl1,bdl2,mt1,mt_br,branch_list,inward, mt_list, region_list,bdl_list, t, natural = natural, origin = origin, branching = branching))
            if self.branch_pos[pt_i] == 0:
                self.branchn[pt_i] = [(len(branch_list)-1)] #TODO revisit in geodesic case
            else:    
                self.branchn[pt_i].append(len(branch_list)-1)
        else:
            self.branch_pt.append(pt)
            # if not (branching and inward): #if it's not the special branched nucleation case
            pos = arc_pos(pt, self.pol, self.seg, self.seg_dist)
            self.branch_pos.append(pos)
            branch_list.append(branch(bdl1,bdl2,mt1,mt_br,branch_list,inward, mt_list, region_list,bdl_list, t, natural = natural, origin = origin, branching = branching))
            self.branchn.append([len(branch_list)-1])
        if not (branching and inward):
            assert len(self.branch_pos) == len(self.branch_pt)
    def new_branch_events(self, event_list, mt_list, pt, t, bdl_list, branch_list, mt_not = mt_none):
        '''
        Calculate new branch events from a new branch.
        TODO: non-geodesic cases allows us to ignore some of these events (never two-branching)
        Parameters
        ----------
        event_list : List of event class
            Event list.
        mt_list : List of mt class
            MT class.
        pt : List of floats
            Pt of branch.
        t : Float
            Time of creation.
        bdl_list : List of bundle class
            Bdl list.
        mt_not : mt class, optional
            Newly entrained MT, this will not create a new event w/ new branch. The default is mt_none.

        Returns
        -------
        None.

        '''
        br_idx = self.branch_pt.index(pt) #determine index of newest branch pt on bdl
        b_pos = self.branch_pos[br_idx] #its arc-position
        branch_bool = False #if there are two entrained branches, branch crossing is guaranteed
        if len(self.branchn[br_idx]) > 1:
            assert len(self.branchn[br_idx]) == 2
            hand1 = branch_list[self.branchn[br_idx][0]].side
            hand2 = branch_list[self.branchn[br_idx][1]].side
            assert (hand1 == 'east' and hand2 == 'west') or (hand2 == 'east' and hand1 == 'west') #TODO in geodesic case only
            branch_bool = True
            #assert self.branchn[br_idx] == 1 #TODO what is this for??? not needed!
        br = branch_list[self.branchn[br_idx][0]]
        br_pol = None
        if br.branch_pol == 'north':
            br_pol = self.pol
        else:
            br_pol = -1*self.pol
        br_in = br.branch_in #whether branch was going in or out
        bdl2_no = br.branch_bdln #self.branch_bdl[br_idx] #bdl of branch
        
        for i in range(len(self.mts)): #examine when other mts hit this pt
            mt_no = self.mts[i] #mt number
            mt = mt_list[mt_no] #mt
            if mt_no != mt_not.number: #exclude mt_not
                if mt.grow: #only growing mts can hit branches
                    start = self.start_pos[i]
                    current = None
                    dists = mt.seg_dist
                    total_d = 0 #total dist grown
                    for d in range(len(dists)):
                        total_d += dists[d] #add segment distances
                    if mt.grow: #if growing, add current tip length
                        total_d += t-mt.update_t[-1]
                    # elif not mt.grow and not mt.hit_bdry: #if shrinking, subtract
                    #     total_d -= v_s*(t-mt.update_t[-1])
                    if self.rel_polarity[i] > 0: #if growing in + dir of coord system
                        current = start + total_d
                        if mt.grow and current < b_pos:# and (((br_pol<0 and br_in) or (br_pol>0 and not br_in)) or branch_bool): #growing to branch pt, check it's outside of fork
                            dt = b_pos - current
                            event_list.append(event(mt,mt_none,t+dt,pt,'cross_br', calc_t=t))
                            event_list[-1].bdl2 = bdl2_no
                    else:
                        current = start - total_d
                        if mt.grow and current > b_pos:# and (((br_pol>0 and br_in) or (br_pol<0 and not br_in)) or branch_bool): #growing to cross
                            dt = current - b_pos
                            event_list.append(event(mt,mt_none,t+dt,pt,'cross_br', calc_t=t))
                            event_list[-1].bdl2 = bdl2_no
            else: #the updated mt is on the bdl, check its intersection w/ other branches
                if len(self.branch_pos) > 1: #if there are branches other than the new one
                    assert mt.grow #should be growing, only happens on crossovers
                    current = arc_pos(pt, self.pol, self.seg, self.seg_dist)
                    for j in range(len(self.branch_pt)): #check if it collides w/ other branches
                        if j != br_idx: #not the newest branch though
                            pos = self.branch_pos[j] #position of cross
                            #TODO will we need branch geo soon?
                            # br_pol = self.branch_pol[j] #redefine for OTHER branches
                            # br_in = self.branch_in[j]
                            br = branch_list[self.branchn[j][0]]
                            branch_bdl = br.branch_bdln
                            if self.rel_polarity[i] > 0 and current < pos:# and (((br_pol<0 and br_in) or (br_pol>0 and not br_in)) or branch_bool): #mt is growing from behind
                                assert self.branch_pt[j] != pt #shouldn't be the current one
                                dt = pos - current
                                event_list.append(event(mt,mt_none,t+dt,self.branch_pt[j],'cross_br', calc_t=t))
                                event_list[-1].bdl2 = branch_bdl
                            elif self.rel_polarity[i] < 0 and current > pos:# and (((br_pol>0 and br_in) or (br_pol<0 and not br_in)) or branch_bool):
                                assert self.branch_pt[j] != pt #shouldn't be the current one
                                dt = current - pos
                                event_list.append(event(mt,mt_none,t+dt,self.branch_pt[j],'cross_br', calc_t=t))
                                event_list[-1].bdl2 = branch_bdl
    def branch_bdl_event(self, mt, event_list, branch_list):
        '''
        When an MT on the bdl is modified, recalculate possible branch events.

        Parameters
        ----------
        mt : mt class
            MT of interest.
        event_list : List of event class
            Event list.

        Returns
        -------
        None.

        '''
        mt_idx = self.mts.index(mt.number) #idx for position
        pol = self.rel_polarity[mt_idx]
        current = self.start_pos[mt_idx]
        if len(mt.seg) > 1: #if there are more segs, must be rescue event and current pos isn't 0
            current = arc_pos(mt.seg[-1], self.pol, self.seg, self.seg_dist)
        t = mt.update_t[-1]
        for i in range(len(self.branch_pos)): #check crossover events
            pos2 = self.branch_pos[i]
            pt = self.branch_pt[i]
            branch_bool = False #if there are two entrained branches, branch crossing is guaranteed
            if len(self.branchn[i]) > 1:
                assert len(self.branchn[i]) == 2
                hand1 = branch_list[self.branchn[i][0]].side
                hand2 = branch_list[self.branchn[i][1]].side
                assert (hand1 == 'east' and hand2 == 'west') or (hand2 == 'east' and hand1 == 'west') #TODO in geodesic case only
                branch_bool = True
            br = branch_list[self.branchn[i][0]]
            br_pol = None
            if br.branch_pol == 'north':
                br_pol = self.pol
            else:
                br_pol = -1*self.pol
            br_in = br.branch_in #whether branch was going in or out
            bdl2_no = br.branch_bdln #self.branch_bdl[br_idx] #bdl of branch
            if pol > 0 and current < pos2:# and (((br_pol<0 and br_in) or (br_pol>0 and not br_in)) or branch_bool):
                dt = pos2 - current
                event_list.append(event(mt,mt_none,t+dt,pt,'cross_br', calc_t=t))
                event_list[-1].bdl2 = bdl2_no
            elif pol < 0 and current > pos2:# and (((br_pol>0 and br_in) or (br_pol<0 and not br_in)) or branch_bool):
                dt = current - pos2
                event_list.append(event(mt,mt_none,t+dt,pt,'cross_br', calc_t=t))
                event_list[-1].bdl2 = bdl2_no
    def overtake_bdl_event(self, mt1, pt, t, mt_list, event_list, recalc_grow=False):
        '''
        Calculate whether current mt overtakes any other on the bdl, append to event_list.

        Parameters
        ----------
        mt1 : mt class
            MT of interest.
        pt : List of floats
            Pt of current mt.
        t : Float
            Current time.
        mt_list : List mt class
            MT list.
        event_list : List of event class
            Event list.
        recalc_grow: bool
            Indicates special case where there is new growth AND tread,
            requires two calls. Specific to parallel nucleation.

        Returns
        -------
        None.

        '''
        if deflect_on: #if no deflection, do nothing!
            if len(self.mts) > 1: #if there are other mts to worry about
                # print(mt1.number)
                pos1 = arc_pos(pt, self.pol, self.seg, self.seg_dist)
                # pos1 = dist(pt,self.seg[0]) #position of mt1
                # if pt[0] < self.seg[0][0]:
                #     pos1 *= -1 #change sign if needed
                k = self.mts.index(mt1.number) #for finding polarity of mt1
                pol1 = self.rel_polarity[k]
                for i in range(len(self.mts)): #examine when other mts hit this pt
                    mt_no = self.mts[i] #mt number
                    pol2 = self.rel_polarity[i] #polarity
                    mt = mt_list[mt_no] #mt
                    if mt_no != mt1.number: #exclude mt1
                        start = self.start_pos[i]
                        current = None
                        dists = mt.seg_dist
                        total_d = 0 #total dist grown
                        for d in range(len(dists)):
                            total_d += dists[d] #add segment distances
                        if mt.grow: #if growing, add current tip length
                            total_d += t-mt.update_t[-1]
                        elif not mt.grow and not mt.hit_bdry: #if shrinking, subtract
                            total_d -= v_s*(t-mt.update_t[-1])
                        if mt1.grow and (mt1.tread_t != t or recalc_grow):
                            if pol1 > 0: #mt1 growing in + dir
                                if pol2 > 0 and (not mt.grow and not mt.hit_bdry): #if other mt shriking. No 1catch2_m since v_t<v_g
                                    current = start + total_d
                                    dt = (current-pos1)/(v_s+1)
                                    tread_bool = True #for tread cases, check when it disappears
                                    if mt.tread:
                                        tread_d = total_d - (t-mt.tread_t)*mt.vt
                                        dt2 = tread_d/(mt.vt+v_s)
                                        tread_bool = (dt2 > dt) #if it takes longer to disappear than to catch up
                                    if current > pos1 and v_s*dt < total_d and tread_bool: #TODO: do I need 2*dt part or just tread_bool?
                                        pos2 = pos1 + pol1*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,pt2,'1catch2',calc_t=t))
                                elif pol2 > 0 and (mt.hit_bdry and mt.ext_mt == None): #paused plus end
                                    current = start + total_d
                                    dt = (current-pos1)#/(1)
                                    tread_bool = True #for tread cases, check when it disappears
                                    if mt.tread:
                                        tread_d = total_d - (t-mt.tread_t)*mt.vt
                                        dt2 = tread_d/(mt.vt)
                                        tread_bool = (dt2 > dt) #if it takes longer to disappear than to catch up
                                    if current > pos1 and tread_bool:
                                        pos2 = pos1 + pol1*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,pt2,'1catch2',calc_t=t))
                                elif pol2 < 0 and (not mt.grow and not mt.hit_bdry): #if other mt shriking other dir
                                    current = start - total_d
                                    dt = (pos1-current)/(v_s-1)
                                    tread_bool = True #whether disap will happen after catchup
                                    if mt.tread:
                                        pos_n = start - (t-mt.tread_t)*mt.vt #pos of minus end
                                        dt3 = (pos_n-current)/(mt.vt+v_s) #time till disap
                                        tread_bool = (dt3>dt)
                                        if pos1 < pos_n: # can catch up
                                            dt2 = (pos_n-pos1)/(1+mt.vt)
                                            # tread_bool2 = True#(dt3>dt2) #whether disap will happen after catchup_m
                                            # if current < pos1:# and 2*dt2 < total_d and tread_bool2: #catch up the minus end
                                            pos2 = pos1 + pol1*dt2 #new arc position
                                            pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                            event_list.append(event(mt1,mt,t+dt2,pt2,'1catch2_m',calc_t=t))
                                    if current < pos1 and v_s*dt < total_d and tread_bool: #can only catch up in this case
                                        posp = pos1 + pol1*dt #new arc position
                                        ptp = arc_pt(posp, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,ptp,'1catch2',calc_t=t))
                                elif pol2 < 0 and mt.hit_bdry and mt.tread: #if other mt tread other dir (stationary plus)
                                    pos_n = start - (t-mt.tread_t)*mt.vt #pos of minus end
                                    if pos1 < pos_n: # can catch up
                                        dt = (pos_n-pos1)/(1+mt.vt)
                                        # if mt.vt*dt < total_d: #can only catch up in this case
                                        pos2 = pos1 + pol1*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,pt2,'1catch2_m',calc_t=t))
                                elif pol2 < 0 and not mt.hit_bdry and mt.tread: #if other mt tread other dir (growing plus)
                                    pos_n = start - (t-mt.tread_t)*mt.vt #pos of minus end
                                    if pos1 < pos_n: # can catch up
                                        dt = (pos_n-pos1)/(1+mt.vt)
                                        pos2 = pos1 + pol1*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,pt2,'1catch2_m',calc_t=t))
                            if pol1 < 0: #mt1 growing in - dir
                                if pol2 > 0 and (not mt.grow and not mt.hit_bdry): #if other mt shriking
                                    current = start + total_d
                                    dt = (current-pos1)/(v_s-1)
                                    tread_bool = True
                                    if mt.tread:
                                        pos_n = start + (t-mt.tread_t)*mt.vt #pos of minus end
                                        dt3 = (current-pos_n)/(v_s+mt.vt)
                                        tread_bool = (dt3>dt)
                                        if pos1 > pos_n: # can catch up
                                            dt2 = (-pos_n+pos1)/(1+mt.vt)
                                            # tread_bool2 = (dt3>dt2)
                                            # if current > pos1:# and 2*dt2 < total_d and tread_bool2: #can only catch up in this case
                                            pos2 = pos1 + pol1*dt2 #new arc position
                                            pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                            event_list.append(event(mt1,mt,t+dt2,pt2,'1catch2_m',calc_t=t))
                                    if current > pos1 and v_s*dt < total_d and tread_bool: #can only catch up in this case
                                        posp = pos1 + pol1*dt #new arc position
                                        ptp = arc_pt(posp, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,ptp,'1catch2',calc_t=t))
                                elif pol2 > 0 and mt.hit_bdry and mt.tread: #if other mt tread other dir (stationary plus)
                                    pos_n = start + (t-mt.tread_t)*mt.vt #pos of minus end
                                    if pos1 > pos_n: # can catch up
                                        dt = (-pos_n+pos1)/(1+mt.vt)
                                        # if mt.vt*dt < total_d: #can only catch up in this case
                                        pos2 = pos1 + pol1*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,pt2,'1catch2_m',calc_t=t))
                                elif pol2 > 0 and not mt.hit_bdry and mt.tread: #if other mt tread other dir (growing plus)
                                    pos_n = start + (t-mt.tread_t)*mt.vt #pos of minus end
                                    if pos1 > pos_n: # can catch up
                                        dt = (-pos_n+pos1)/(1+mt.vt)
                                        pos2 = pos1 + pol1*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,pt2,'1catch2_m',calc_t=t))
                                elif pol2 < 0 and (not mt.grow and not mt.hit_bdry): #if other mt shriking other dir, v_t<v_g
                                    current = start - total_d
                                    dt = (pos1-current)/(1+v_s)
                                    tread_bool = True #for tread cases
                                    if mt.tread:
                                        tread_d = total_d - (t-mt.tread_t)*mt.vt
                                        dt2 = tread_d/(mt.vt+v_s)
                                        tread_bool = (dt2 > dt) #if it takes longer to disappear than to catch up
                                    if current < pos1 and v_s*dt < total_d and tread_bool: #can only catch up in this case
                                        pos2 = pos1 + pol1*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,pt2,'1catch2',calc_t=t))
                                elif pol2 < 0 and (mt.hit_bdry and mt.ext_mt == None): #paused plus end
                                    current = start - total_d
                                    dt = pos1-current#/(1)
                                    tread_bool = True #for tread cases, check when it disappears
                                    if mt.tread:
                                        tread_d = total_d - (t-mt.tread_t)*mt.vt
                                        dt2 = tread_d/(mt.vt)
                                        tread_bool = (dt2 > dt) #if it takes longer to disappear than to catch up
                                    if current < pos1 and tread_bool:
                                        pos2 = pos1 + pol1*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt1,mt,t+dt,pt2,'1catch2',calc_t=t))
                        elif mt1.tread_t == t: #invoked at time of tread start, no need to recalculate what happens at plus end, simpler version of above case
                            dists1 = mt1.seg_dist
                            total_d1 = 0 #total dist grown for mt1, see if it shrinks before overtake
                            for d in range(len(dists1)):
                                total_d1 += dists1[d] #add segment distances
                            if not mt1.grow and not mt1.hit_bdry:
                                total_d1 -= v_s*(t-mt1.update_t[-1])
                            if pol1 > 0: #mt1 grows in + dir
                                if pol2 < 0 and mt.grow: #if other mt shriking other dir
                                    current = start - total_d
                                    dt = (-pos1+current)/(mt1.vt+1)
                                    tread_bool = True
                                    if current > pos1 and tread_bool: #can only catch up in this case
                                        pos2 = current + pol2*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt,mt1,t+dt,pt2,'1catch2_m',calc_t=t))
                            if pol1 < 0: #mt1 grows in - dir
                                if pol2 > 0 and mt.grow: #if other mt shriking
                                    current = start + total_d
                                    dt = (-current+pos1)/(mt1.vt+1)
                                    tread_bool = True
                                    if current < pos1 and tread_bool: #can only catch up in this case
                                        pos2 = current + pol2*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt,mt1,t+dt,pt2,'1catch2_m',calc_t=t))
                        elif not mt1.grow and mt1.hit_bdry:
                            assert mt1.ext_mt == None
                            if pol1 > 0: #mt1 grows in + dir
                                if pol2 > 0 and mt.grow: #if other mt shriking
                                    current = start + total_d
                                    if current < pos1: #can only catch up in this case
                                        dt = (pos1-current)#/(1)
                                        pt2 = [pt[0],pt[1]]
                                        event_list.append(event(mt,mt1,t+dt,pt2,'1catch2',calc_t=t))
                            if pol1 < 0: #mt1 grows in - dir
                                if pol2 < 0 and mt.grow: #if other mt shriking other dir
                                    current = start - total_d
                                    if current > pos1: #can only catch up in this case
                                        dt = (-pos1+current)#/(+1)
                                        pt2 = [pt[0],pt[1]]
                                        event_list.append(event(mt,mt1,t+dt,pt2,'1catch2',calc_t=t))
                        else: #mt1 is shrinking, compare w/ growing ones
                            dists1 = mt1.seg_dist
                            total_d1 = 0 #total dist grown for mt1, see if it shrinks before overtake
                            for d in range(len(dists1)):
                                total_d1 += dists1[d] #add segment distances
                            total_d1 -= v_s*(t-mt1.update_t[-1])
                            if pol1 > 0: #mt1 grows in + dir
                                if pol2 > 0 and mt.grow: #if other mt shriking
                                    current = start + total_d
                                    dt = (pos1-current)/(1+v_s)
                                    if current < pos1 and v_s*dt < total_d1: #can only catch up in this case
                                        pos2 = current + pol2*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt,mt1,t+dt,pt2,'1catch2',calc_t=t))
                                elif pol2 < 0 and mt.grow: #if other mt shriking other dir
                                    current = start - total_d
                                    dt = (pos1-current)/(v_s-1)
                                    if current < pos1 and v_s*dt < total_d1: #can only catch up in this case
                                        pos2 = current + pol2*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt,mt1,t+dt,pt2,'1catch2',calc_t=t))
                            if pol1 < 0: #mt1 grows in - dir
                                if pol2 > 0 and mt.grow: #if other mt shriking
                                    current = start + total_d
                                    dt = (current-pos1)/(v_s-1)
                                    if current > pos1 and v_s*dt < total_d1: #can only catch up in this case
                                        pos2 = current + pol2*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt,mt1,t+dt,pt2,'1catch2',calc_t=t))
                                elif pol2 < 0 and mt.grow: #if other mt shriking other dir
                                    current = start - total_d
                                    dt = (-pos1+current)/(v_s+1)
                                    if current > pos1 and v_s*dt < total_d1: #can only catch up in this case
                                        pos2 = current + pol2*dt #new arc position
                                        pt2 = arc_pt(pos2, self.seg, self.angle, self.seg_dist)
                                        event_list.append(event(mt,mt1,t+dt,pt2,'1catch2',calc_t=t))
    def new_bdl_deflect(self,event_list,mt_list,bdl_list,region_list,pt,t,mt_not,):
        '''
        Just added new vertex, inform that other mts must turn on new vertices.

        Parameters
        ----------
        event_list : List of event class
            Event list.
        mt_list : List of mt class
            MT list.
        bdl_list : List of bdl class
            Bdl list.
        region_list : List of region_traj
            Traj list.
        pt : List of floats
            Pt of vertex.
        t : Float
            Current time.
        mt_not : mt class
            Leading MT, can ignore in below calculations

        Returns
        -------
        None.

        '''
        seg_idx = self.seg.index(pt) #determine index of crossed pt on bdl
        for i in range(len(self.mts)): #examine when other mts hit this pt
            mt_no = self.mts[i] #mt number
            mt = mt_list[mt_no] #mt
            if mt_no != mt_not.number and mt.grow and self.rel_polarity[i]==self.pol: #exclude mt_not and mts not currently growing toward new vertex
                start = self.start_pos[i]
                dists = mt.seg_dist
                abs_pos = None #absolute arc position
                total_d = 0 #total dist grown
                for d in range(len(dists)):
                    total_d += dists[d] #add segment distances
                total_d += t-mt.update_t[-1] #if growing, add current tip length
                if self.rel_polarity[i] > 0: #if growing in + dir of coord system
                    abs_pos = abs(start + total_d)
                else:
                    abs_pos = abs(start - total_d)
                cumsum = np.append([0],np.cumsum(self.seg_dist)) #cumulative sums
                j = 0
                while abs_pos >= cumsum[j]: #find index within cumsum
                    j+=1
                    if j > len(cumsum)-1:
                        break
                j -= 1 #jth seg
                if j == seg_idx -1: #if mt is passed prev vertex already
                    if (self.pol > 0 and self.ext_bdl is not None) or (self.pol < 0 and self.prev_bdl is not None): #if vertex is bdry pt
                        # bdry_res = inter_r_bdry(mt,mt_list, bdl_list) #find intersection info
                        bdry_res = inter_r_bdry2(mt, mt_list, bdl_list, region_list)
                        next_time = bdry_res[0] + mt.update_t[-1] #time of collision
                        event_list.append(event(mt,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
                    else: #vertex is actual vertex
                        dt = cumsum[j+1] - abs_pos
                        event_list.append(event(mt,mt_none,t+dt,pt,'follow_bdl',calc_t=t))
    def deflect_bdl_event(self,event_list,mt_list,bdl_list,region_list,t,mt,pt, bypass = False):
        '''
        New mt growing on bdl, find next vertex to follow.

        Parameters
        ----------
        event_list : List of event class
            Event list.
        mt_list : List of mt class
            MT list.
        bdl_list : List of bdl class
            Bdl list.
        region_list : List of region_traj
            Traj list.
        t : Float
            Current time.
        mt : mt class
            MT of interest.
        pt : List of floats
            Current pt.
        bypass: Bool
            Bypasses certain bool statements
        Returns
        -------
        None.

        '''
        assert len(self.mts) > 0 #should have mts to follow
        mt_idx = self.mts.index(mt.number)
        start = self.start_pos[mt_idx]
        dists = mt.seg_dist
        abs_pos = None #absolute arc position
        total_d = 0 #total dist grown
        for d in range(len(dists)):
            total_d += dists[d] #add segment distances
        if mt.grow: #if growing, add current tip length
            total_d += t-mt.update_t[-1]
        elif not mt.grow and not mt.hit_bdry: #if shrinking, subtract
            total_d -= v_s*(t-mt.update_t[-1])
        if self.rel_polarity[mt_idx] > 0: #if growing in + dir of coord system
            abs_pos = abs(start + total_d)
        else:
            abs_pos = abs(start - total_d)
        cumsum = np.append([0],np.cumsum(self.seg_dist)) #cumulative sums
        j = 0
        while abs_pos >= cumsum[j]: #find index within cumsum, equality important for j=0 case
            j+=1
            if abs_pos == cumsum[j-1] and j != 1:
                j -= 1 #special case when cross_br -> follow_br since pt right on seg pt
                break
            if j > len(cumsum)-1:
                break
        j -= 1 #jth seg
        if self.pol==self.rel_polarity[mt_idx]:# > 0: #if mt growing in same dir of bdl
            if j+1 < len(self.seg): #if next vertex exists
                if j+1 == len(self.seg)-1 and ((self.pol > 0 and self.ext_bdl is not None) or (self.pol < 0 and self.prev_bdl is not None)): #next vertex is bdry
                    # bdry_res = inter_r_bdry(mt,mt_list, bdl_list) #find intersection info
                    bdry_res = inter_r_bdry2(mt,mt_list,bdl_list,region_list)
                    next_time = bdry_res[0] + mt.update_t[-1] #time of collision
                    event_list.append(event(mt,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t = t))
                else: #next vertex is not bdry
                    pt = self.seg[j+1] #next vertex
                    assert pt not in self.cross_pt #this pt possibly included in cross_bdl event
                    dt = cumsum[j+1] - abs_pos #dist to next vertex
                    event_list.append(event(mt,mt_none,t+dt,pt,'follow_bdl',calc_t=t))
            else:
                #TODO still need to worry about this case when the first mt hits bdry and this can't be updated causing numerical errors
                if len(self.mts) == 1 and deflect_on and (abs(mt.angle[-1] - pi/2)>vtol and abs(mt.angle[-1]-3*pi/2)>vtol): #ONLY FOR FOLLOW_BR EVENT, no other vertices and only one mt, most recent one
                    if mt.update_t[-1] == t:
                        assert self.branch_pt.index(pt) == 0 #if len > 2, other mt will change before this, no need to calculate for this mt
                # bdry_res = inter_r_bdry(mt,mt_list, bdl_list, free=True) #as if it's a leader mt
                bdry_res = inter_r_bdry2(mt,mt_list, bdl_list, region_list, free=mt.free)
                if bdry_res[2] != 'deflect' or len(self.mts) == 1 or bypass: #if it's deflect and mts>1, the mt ahead will update and call this fn
                    next_time = bdry_res[0] + mt.update_t[-1] #bypass is only used in cross event since the above statement might ignore certain events
                    event_list.append(event(mt,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
        else:
            if j == 0 and ((self.pol > 0 and self.prev_bdl is not None) or (self.pol < 0 and self.ext_bdl is not None)): #vertex is bdry pt
                # bdry_res = inter_r_bdry(mt,mt_list, bdl_list) #find intersection info TODO THIS PART IS SPECIAL, CAN BE IN OTHER DIR
                bdry_res = inter_r_bdry2(mt,mt_list, bdl_list, region_list, to_wall = True)
                next_time = bdry_res[0] + mt.update_t[-1] #time of collision
                event_list.append(event(mt,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t = t))
            elif j != 0:
                pt = self.seg[j] #prev vertex
                assert pt not in self.cross_pt #this pt possibly included in cross_bdl event
                dt = abs_pos - cumsum[j]
                if dt <= 0:
                    print(abs_pos,cumsum[j], dt, j)
                    assert dt > 0
                event_list.append(event(mt,mt_none,t+dt,pt,'follow_bdl',calc_t=t))
    def deflect_bdl_event_v(self,event_list,mt_list,bdl_list,region_list,t,mt,pt):
        '''
        Last event was MT hitting vertex, find next vertex to follow.

        Parameters
        ----------
        event_list : List of event class
            Event list.
        mt_list : List of mt class
            MT list.
        bdl_list : List of bdl class
            Bdl list.
        region_list : List of region_traj
            Traj list.
        t : Float
            Current time.
        mt : mt class
            MT of interest.
        pt : List of floats
            Current pt.

        Returns
        -------
        None.

        '''
        j = self.seg.index(pt) #seg index
        mt_idx = self.mts.index(mt.number) #mt index
        if self.pol==self.rel_polarity[mt_idx]:# > 0: #if mt growing in same dir of bdl
            if j+1 < len(self.seg): #if next vertex exists
                if j+1 == len(self.seg)-1 and ((self.pol > 0 and self.ext_bdl is not None) or (self.pol < 0 and self.prev_bdl is not None)): #next vertex is bdry
                    # bdry_res = inter_r_bdry(mt,mt_list, bdl_list) #find intersection info
                    bdry_res = inter_r_bdry2(mt,mt_list, bdl_list, region_list)
                    next_time = bdry_res[0] + mt.update_t[-1] #time of collision
                    event_list.append(event(mt,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t = t))
                else: #next vertex is not bdry
                    pt = self.seg[j+1] #next vertex
                    assert pt not in self.cross_pt #possibly in cross_bdl event
                    dt = self.seg_dist[j] #dist to next vertex
                    event_list.append(event(mt,mt_none,t+dt,pt,'follow_bdl',calc_t = t))
            else:
                assert j+1 == len(self.seg) #must be last seg
                # bdry_res = inter_r_bdry(mt,mt_list, bdl_list) #find intersection info
                bdry_res = inter_r_bdry2(mt,mt_list, bdl_list, region_list)
                if bdry_res[2] != 'deflect': #only use this for possible bdry collisions
                    next_time = bdry_res[0] + mt.update_t[-1] #time of collision
                    event_list.append(event(mt,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
        else:#TODO still need to worry about this case when the first mt hits bdry and this can't be updated causing numerical errors
            if j-1 == 0 and ((self.pol > 0 and self.prev_bdl is not None) or (self.pol < 0 and self.ext_bdl is not None)): #vertex is bdry pt
                # bdry_res = inter_r_bdry(mt,mt_list, bdl_list) #find intersection info TODO THIS IS SPECIAL CASE
                bdry_res = inter_r_bdry2(mt,mt_list, bdl_list, region_list,to_wall=True)
                next_time = bdry_res[0] + mt.update_t[-1] #time of collision
                event_list.append(event(mt,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t = t))
            elif j-1 != 0:
                pt = self.seg[j-1] #prev vertex
                assert pt not in self.cross_pt #possiply in cross_bld event
                dt = self.seg_dist[j-1] #distance to prev vertex
                event_list.append(event(mt,mt_none,t+dt,pt,'follow_bdl',calc_t = t))
    def add_cross_bdl(self, pt, bdl):
        '''
        Add new crossing to this bdl.

        Parameters
        ----------
        pt : List of floats
            Cross pt.
        bdl : bundle class
            Crossing bdl.

        Returns
        -------
        None.

        '''
        if deflect_on: #w deflection,crossovers are always new
            # assert pt not in self.cross_pt #TODO nearly vertical mts raise this exception, need to look into it
            if pt in self.cross_pt: #overwrite prev crossover instead of adding new
                pt_i = self.cross_pt.index(pt)
                assert len(self.cross_mts[pt_i]) == 0 #crossover can exist by cannot be occupied
                del self.cross_bdl[pt_i] #to be consisdent w/ alg, delete and re-add rather than reuse current info
                del self.cross_pt[pt_i]
                del self.cross_mts[pt_i]
                del self.cross_pos[pt_i]
            assert bdl.number not in self.cross_bdl
            self.cross_pt.append(pt)
            self.cross_bdl.append(bdl.number)
            pos = arc_pos(pt, self.pol, self.seg, self.seg_dist)
            self.cross_pos.append(pos)
        else: #no deflection, crossover may already exists
            if pt in self.cross_pt: #overwrite prev crossover instead of adding new
                pt_i = self.cross_pt.index(pt)
                self.cross_bdl[pt_i] = bdl.number
            else: #create new
                assert bdl.number not in self.cross_bdl
                self.cross_pt.append(pt)
                self.cross_bdl.append(bdl.number)
                pos = arc_pos(pt, self.pol, self.seg, self.seg_dist)
                self.cross_pos.append(pos)
    def add_mt(self, mt):
        '''
        New mt entrained on bdl.

        Parameters
        ----------
        mt : mt class
            New MT.

        Returns
        -------
        None.

        '''
        mt.bdl = self.number #assign bdl no
        self.mts.append(mt.number)
        pt = mt.seg[0] #point of mt should be its first pt
        pos = arc_pos(pt, self.pol, self.seg, self.seg_dist)
        self.start_pos.append(pos)
        pol = 1 #polarity
        if mt.angle[-1] > pi/2 and mt.angle[-1] < 3*pi/2:
            pol = -1
        self.rel_polarity.append(pol)
    def del_mt(self, mt):
        '''
        Deleting MT from this bdl

        Parameters
        ----------
        mt : mt class
            MT to be deleted.

        Returns
        -------
        None.

        '''
        idx = self.mts.index(mt.number)
        del self.mts[idx]
        del self.rel_polarity[idx]
        del self.start_pos[idx]
        del self.mt_sides[idx]
        # if (mt.prev_mt is not None) and (mt.prev_mt in self.branch): #could be from zippering
        #     self.branch.remove(mt.prev_mt) #prev mt no longer a branch
    def del_cross(self, mt, pt):
        '''
        Deletes crossing (external) mt from pov of current bundle.

        Parameters
        ----------
        mt : mt class
            MT to be deleted.
        pt : List of floats
            Cross pt.

        Returns
        -------
        None.

        '''
        cross_gone = False #whether the bundle is still crossed
        cross_idx = self.cross_bdl.index(mt.bdl) #idx in cross structure
        if mt.number not in self.cross_mts[cross_idx]:
            print(self.cross_mts[cross_idx])
        mt_idx = self.cross_mts[cross_idx].index(mt.number) #idx in mt list
        del self.cross_mts[cross_idx][mt_idx] #get rid of it
        if len(self.cross_mts[cross_idx]) == 0: # no more mts here, get rid of cross info
            # del self.cross_bdl[cross_idx]
            # del self.cross_pt[cross_idx]
            # del self.cross_mts[cross_idx]
            # del self.cross_pos[cross_idx]
            cross_gone = True
        return(cross_gone)
    def del_cross_bdl(self, pt):
        '''
        Deletes crossing bdl from current bdl, only happens when del_cross returns true

        Parameters
        ----------
        pt : List of floats
            Cross pt.

        Returns
        -------
        None.

        '''
        cross_idx = self.cross_pt.index(pt)
        del self.cross_bdl[cross_idx]
        del self.cross_pt[cross_idx]
        del self.cross_mts[cross_idx]
        del self.cross_pos[cross_idx]
    def mt_overlap(self, mt_list, pt, t, mt2_n = None, mt_none = [],branch_cross=False):
        '''
        Determine how many MTs that occupy this pt.

        Parameters
        ----------
        mt_list : List of mt class
            MT list.
        pt : List of floats
            Pt of interest.
        t : Float
            Current time.
        mt2_n : Int, optional
            Newly entrained MT ID. The default is None.
        mt_none : List of ints, optional
            MT IDs to ignore. The default is [].
        branch_crass : whether this is called at a branch cross event

        Returns
        -------
        None.

        '''
        pos = arc_pos(pt, self.pol, self.seg, self.seg_dist)
        # pos = dist(pt, self.seg[0])
        # if pt[0] < self.seg[0][0]: #if x position behind, position is negative
        #     pos *= -1
        mt_in = [] #list of mts in the interval
        for i in range(len(self.mts)): #examine when other mts hit this pt
            mt_idx = self.mts[i] #mt index
            if mt_idx not in mt_none: #if there's an excluded mt
                mt = mt_list[mt_idx] #mt
                start = self.start_pos[i]
                special_case = False #special case where we're at a branch where there already exists a natural branch ahead but cannot resolve normally (mt starts exactly at the branch)
                #happens only when there are branch mts on both sides which creates a "crossing", but the crossing doesn't actually have
                #mts that traverse it, hence unable to resolve the mts' presence
                specialer = False #case where the pt is exactly at the mt tip. sorry, lots of cases
                if branch_cross: #to save compute time, only call when at branch crossing
                    if pt in self.branch_pt:
                        if mt.seg[0] == pt:
                            mt1n = mt_none[0] #in the case we're interested in, these mts must have same polarity
                            mt1_idx = self.mts.index(mt1n)
                            mt1_pol = self.rel_polarity[mt1_idx]
                            if mt1_pol == self.rel_polarity[i]:
                                special_case = True
                        elif mt.seg[-1] == pt:
                            mt1n = mt_none[0] #in othercase, polarity is different
                            mt1_idx = self.mts.index(mt1n)
                            mt1_pol = self.rel_polarity[mt1_idx]
                            if mt1_pol != self.rel_polarity[i]:
                                special_case = True
                                specialer = True
                if not mt.grow and mt.hit_bdry and (mt.seg[-1] not in [self.ext_pt, self.prev_pt]): #in the case where the pt is given
                    current = None
                    stall = mt.seg[-1] #last point on mt
                    if stall in self.branch_pt: #reuse arc-pos for precision
                        pt_i = self.branch_pt.index(stall)
                        current = self.branch_pos[pt_i]
                    else: #not a branch, due to pause state
                        if not no_bdl_id:#in this case, the branch point should not exist
                            assert mt.ext_mt == None
                        current = arc_pos(stall, self.pol, self.seg, self.seg_dist)
                    tread_bool = True
                    if self.rel_polarity[i] > 0 and (start < pos or special_case): #if growing in + dir of coord system
                        if mt.tread:
                            neg = start + (t-mt.tread_t)*mt.vt
                            if neg > pos:
                                tread_bool = False
                        if (pos < current or (pos==current and specialer)) and tread_bool: #if passed cross point
                            mt_in.append(mt.number)
                    elif self.rel_polarity[i] < 0 and (start > pos or special_case):
                        if mt.tread:
                            neg = start - (t-mt.tread_t)*mt.vt
                            if neg < pos:
                                tread_bool = False
                        if (pos > current or (pos==current and specialer)) and tread_bool: #overtaken
                            mt_in.append(mt.number)
                else:
                    dists = mt.seg_dist
                    total_d = 0 #total dist grown
                    tread_bool = True
                    for d in range(len(dists)):
                        total_d += dists[d] #add segment distances
                    if mt.grow: #if growing, add current tip length
                        total_d += t-mt.update_t[-1]
                    elif not mt.grow and not mt.hit_bdry: #if shrinking, subtract
                        total_d -= v_s*(t-mt.update_t[-1])
                    if self.rel_polarity[i] > 0 and (start < pos or special_case): #if growing in + dir of coord system
                        current = start + total_d
                        if mt.tread:
                            neg = start + (t-mt.tread_t)*mt.vt
                            if neg > pos:
                                tread_bool = False
                        if (pos < current or (pos==current and specialer)) and tread_bool: #if passed cross point
                            mt_in.append(mt.number)
                    elif self.rel_polarity[i] < 0 and (start > pos or special_case):
                        current = start - total_d
                        if mt.tread:
                            neg = start - (t-mt.tread_t)*mt.vt
                            if neg < pos:
                                tread_bool = False
                        if (pos > current or (pos==current and specialer)) and tread_bool: #overtaken
                            mt_in.append(mt.number)
                    # td.append(total_d)
        if mt2_n != None:
            assert mt2_n in mt_in
        # if len(mt_in) < 1:
        #     print(pos, starts, td, self.rel_polarity, self.number)
        #     assert len(mt_in) > 0
        return(mt_in)
    def new_cross_events(self, event_list, mt_list, pt, t, bdl_list, mt_not = mt_none): #TODO: FIX THE TREAD STUFF@!!!!!
        '''
        New crossover pt, calculate mt collisions w/ new cross pt.

        Parameters
        ----------
        event_list : List of event class
            Event list.
        mt_list : List of mt class
            MT list.
        pt : List of floats
            Cross pt.
        t : Float
            Current time.
        bdl_list : List of bdl class
            Bdl list.
        mt_not : mt class, optional
            MT to ignore. The default is mt_none.

        Returns
        -------
        None.

        '''
        cross_idx = self.cross_pt.index(pt) #determine index of crossed pt on bdl
        bdl2_no = self.cross_bdl[cross_idx] #bdl that gets intersected
        c_pos = self.cross_pos[cross_idx] #its arc-position
        for i in range(len(self.mts)): #examine when other mts hit this pt
            mt_no = self.mts[i] #mt number
            mt = mt_list[mt_no] #mt
            if mt_no != mt_not.number: #exclude mt_not
                start = self.start_pos[i]
                current = None
                dists = mt.seg_dist
                total_d = 0 #total dist grown
                for d in range(len(dists)):
                    total_d += dists[d] #add segment distances
                if mt.grow: #if growing, add current tip length
                    total_d += t-mt.update_t[-1]
                elif not mt.grow and not mt.hit_bdry: #if shrinking, subtract
                    total_d -= v_s*(t-mt.update_t[-1])
                if self.rel_polarity[i] > 0: #if growing in + dir of coord system
                    if start < c_pos and mt.tread:
                        t2 = (c_pos - start)/mt.vt+mt.tread_t
                        if t2>t:
                            event_list.append(event(mt,mt_none,t2,pt,'uncross_m',calc_t = t))
                            event_list[-1].bdl2 = bdl2_no
                    current = start + total_d
                    if mt.grow and current < c_pos: #growing to cross pt
                        dt = c_pos - current
                        event_list.append(event(mt,mt_none,t+dt,pt,'cross_bdl',calc_t = t))
                        event_list[-1].bdl2 = bdl2_no
                    if current > c_pos and start < c_pos:
                        if not mt.grow and not mt.hit_bdry:# and not mt.tread: #same as original
                            dt = (current - c_pos)/v_s
                            event_list.append(event(mt,mt_none,t+dt,pt,'uncross',calc_t = t))
                            event_list[-1].bdl2 = bdl2_no
                            assert bdl2_no != None
                else:
                    current = start - total_d
                    if start > c_pos and mt.tread:
                        t2 = (start - c_pos)/mt.vt+mt.tread_t
                        if t2>t:
                            event_list.append(event(mt,mt_none,t2,pt,'uncross_m',calc_t = t))
                            event_list[-1].bdl2 = bdl2_no
                    if mt.grow and current > c_pos: #growing to cross
                        dt = current - c_pos
                        event_list.append(event(mt,mt_none,t+dt,pt,'cross_bdl',calc_t = t))
                        event_list[-1].bdl2 = bdl2_no
                    if current < c_pos and start > c_pos:
                        if not mt.grow and not mt.hit_bdry:# and not mt.tread: #same as original
                            dt = (c_pos - current)/v_s
                            event_list.append(event(mt,mt_none,t+dt,pt,'uncross',calc_t = t))
                            event_list[-1].bdl2 = bdl2_no
                            assert bdl2_no != None
            else: #the updated mt is on the bdl
                assert mt.grow #should be growing, only happens on crossovers
                current = arc_pos(pt, self.pol, self.seg, self.seg_dist)
                # current = dist(pt, self.seg[0]) #current position is exactly at updated pt
                # if pt[0] < self.seg[0][0]: #opposisite direction
                #     current *= -1
                for j in range(len(self.cross_pt)): #check if it collides w/ other crossovers
                    pos = self.cross_pos[j] #position of cross
                    if self.cross_pt[j] == pt and mt.tread:
                        pol = self.rel_polarity[i]
                        start = self.start_pos[i]
                        t2 = pol*(pos-start)/mt.vt + mt.tread_t
                        assert t2>t
                        event_list.append(event(mt,mt_none,t2,pt,'uncross_m',calc_t = t))
                        event_list[-1].bdl2 = bdl2_no
    def cross_bdl_event(self, mt, event_list):
        '''
        Newly updated MT, find collision of MT on bdl w/ existing cross positions.

        Parameters
        ----------
        mt : mt class
            Current MT.
        event_list : List of event class
            Event list.

        Returns
        -------
        None.

        '''
        mt_idx = self.mts.index(mt.number) #idx for position
        pol = self.rel_polarity[mt_idx]
        start = self.start_pos[mt_idx]
        current = self.start_pos[mt_idx]
        if len(mt.seg) > 1: #if there are more segs, must be rescue event and current pos isn't 0
            current = arc_pos(mt.seg[-1], self.pol, self.seg, self.seg_dist)
            # current  = dist(mt.seg[-1], self.seg[0])
            # if mt.seg[-1][0] < self.seg[0][0]: #negative
            #     current *= -1
        t = mt.update_t[-1]
        for i in range(len(self.cross_pos)): #check crossover events
            pos2 = self.cross_pos[i]
            pt = self.cross_pt[i]
            bdl2_no = self.cross_bdl[i]
            if pol > 0 and current < pos2:
                dt = pos2 - current
                event_list.append(event(mt,mt_none,t+dt,pt,'cross_bdl',calc_t = t))
                event_list[-1].bdl2 = bdl2_no
            elif pol < 0 and current > pos2:
                dt = current - pos2
                event_list.append(event(mt,mt_none,t+dt,pt,'cross_bdl',calc_t = t))
                event_list[-1].bdl2 = bdl2_no
    def uncross_bdl_event(self, mt, pevent, event_list, bdl_list):
        '''
        MT is shrinking, calculate when it uncrosses.

        Parameters
        ----------
        mt : mt class
            MT of interest.
        pevent : event class
            Current event.
        event_list : List of event class
            Event list.
        bdl_list : List of bundle class
            Bdl list.

        Returns
        -------
        None.

        '''
        t = pevent.t#mt.update_t[-1]
        mt_idx = self.mts.index(mt.number) #get mt's index on bdl
        start = self.start_pos[mt_idx]
        polarity = self.rel_polarity[mt_idx]
        end = None
        # print(self.number, mt.number)
        if pevent.policy == 'catas': #to preserve accuracy, use distance directly from pts
            end = arc_pos(pevent.pt, self.pol, self.seg, self.seg_dist)
            # end = dist(origin,pevent.pt)
            # if pevent.pt[0] < origin[0]:
            #     end *= -1
        else: #doesn't matter in other cases
            dists = np.sum(mt.seg_dist) #total length of mt
            if polarity < 0: #if going in negative dir
                dists *= -1
            end = start+dists
        for i in range(len(self.cross_pt)): #check when mt uncrosses
            cross = self.cross_pos[i] #cross pos
            pt = self.cross_pt[i] #cross pt
            bdl2_no = self.cross_bdl[i]
            if polarity > 0:
                if pevent.policy in ['catas','sp_catastrophe','edge_cat', 'disap' , 'pause_to_shrink']:
                    tread_bool = True
                    if mt.tread: #check if already passed cross
                        minus = start + (t-mt.tread_t)*mt.vt
                        tread_bool = (minus < cross)
                    if start < cross and end > cross and tread_bool:
                        dt = (end-cross)/v_s
                        event_list.append(event(mt,mt_none,t+dt,pt,'uncross',calc_t = t))
                        event_list[-1].bdl2 = bdl2_no
                        assert bdl2_no != None
                    elif start < cross: #check the crossovers beyond, should not be occupied by this mt. Could be, due to bdl degen (to be fixed...)
                        bdl2 = bdl_list[bdl2_no]
                        cross_idx = bdl2.cross_pt.index(pt)
                        if mt.number in bdl2.cross_mts[cross_idx]:
                            idx2 = bdl2.cross_mts[cross_idx].index(mt.number)
                            del bdl2.cross_mts[cross_idx][idx2] #COMMENT
                elif pevent.policy in ['disap_tread','parallel_nucleation']: #TODO: separate cases have the same result, delete one?
                    if (mt.grow or mt.hit_bdry) and (start<cross): #stationary or growing
                        dt = (cross-start)/mt.vt
                        event_list.append(event(mt,mt_none,t+dt,pt,'uncross_m',calc_t = t))
                        event_list[-1].bdl2 = bdl2_no
                    elif (not mt.grow and not mt.hit_bdry) and (start<cross): #catstrophe
                        dt2 = (cross-start)/mt.vt
                        event_list.append(event(mt,mt_none,t+dt2,pt,'uncross_m',calc_t = t))
                        event_list[-1].bdl2 = bdl2_no
            elif polarity < 0:
                if pevent.policy in ['catas','sp_catastrophe','edge_cat', 'disap', 'pause_to_shrink']:
                    tread_bool = True
                    if mt.tread: #check if already past cross
                        minus = start - (t-mt.tread_t)*mt.vt
                        tread_bool = (minus > cross)
                    if start  > cross and end < cross and tread_bool:
                        dt = (cross - end)/v_s
                        event_list.append(event(mt,mt_none,t+dt,pt,'uncross',calc_t = t))
                        event_list[-1].bdl2 = bdl2_no
                        assert bdl2_no != None
                    elif start > cross: #check the crossovers beyond, should not be occupied
                        bdl2 = bdl_list[bdl2_no]
                        cross_idx = bdl2.cross_pt.index(pt)
                        if mt.number in bdl2.cross_mts[cross_idx]:
                            idx2 = bdl2.cross_mts[cross_idx].index(mt.number)
                            del bdl2.cross_mts[cross_idx][idx2] #COMMENT
                elif pevent.policy in ['disap_tread','parallel_nucleation']: #TODO: redundant as above?
                    if (mt.grow or mt.hit_bdry) and (start>cross): #stationary or growing
                        dt = (-cross+start)/mt.vt
                        event_list.append(event(mt,mt_none,t+dt,pt,'uncross_m',calc_t = t))
                        event_list[-1].bdl2 = bdl2_no
                    elif (not mt.grow and not mt.hit_bdry) and (start>cross): #catstrophe
                        dt2 = (-cross+start)/mt.vt
                        event_list.append(event(mt,mt_none,t+dt2,pt,'uncross_m',calc_t = t))
                        event_list[-1].bdl2 = bdl2_no
    def check_step_back(self, mt, t, new_pt): #no bdl case: if the new entrainment pos is behind current physical mt
        #only works if MTs are straight!!!
        mt_idx = self.mts.index(mt.number)
        start = self.start_pos[mt_idx]
        polarity = self.rel_polarity[mt_idx]
        minus = start
        if mt.tread:
            minus += polarity*(t-mt.tread_t)*mt.vt #current minus end
        pt_pos = arc_pos(new_pt, self.pol, self.seg, self.seg_dist)
        inverted = False #whether mt got inverted
        if (polarity < 0 and pt_pos > minus):
            inverted = True
        elif polarity > 0 and pt_pos < minus:
            inverted = True
        return(inverted)
    def step_back(self, mt, t, new_pt, pt, inverted = False): #need to check the step back does not invert or uncross. Find best point if it does
        mt_idx = self.mts.index(mt.number)
        start = self.start_pos[mt_idx]
        polarity = self.rel_polarity[mt_idx]
        minus = start
        if mt.tread:
            minus += polarity*(t-mt.tread_t)*mt.vt #current minus end
        new_pt_pos = arc_pos(new_pt, self.pol, self.seg, self.seg_dist)
        pt_pos = arc_pos(pt, self.pol, self.seg, self.seg_dist)
        initial_pos = new_pt_pos #initial pos of step back
        if inverted:
            initial_pos = minus + (pt_pos-minus)/2 #take midpoint so that it's still on the mt
            assert polarity*initial_pos > polarity*minus
        mt_cross_pos = [] #cross_pos occupied by this mt
        for cpos in self.cross_pos: #check among cross_pos whether mt occupies it
            if polarity*cpos > polarity*minus and polarity*cpos < polarity*pt_pos:
                mt_cross_pos.append(cpos)
        if len(mt_cross_pos) > 0: #check cross positions are skipped because of this
            c_pos = None
            if polarity > 0:
                c_pos = max(mt_cross_pos)
            else:
                c_pos = min(mt_cross_pos)
            if polarity*initial_pos < polarity*c_pos:
                initial_pos = c_pos + (pt_pos - c_pos)/2 #take midpoint
        if initial_pos != new_pt_pos: #find new point if it changed
            new_pt = arc_pt(initial_pos, self.seg, self.angle, self.seg_dist)
        return(new_pt)
    def mt_overlap_parallel_nuc(self, mt_list, pt, t, ref_th, meta_th, branching_th, mt2_n, parallel = False):
        '''
        Determine how many MTs that occupy this pt, almost same as mt_overlap_parallel_nuc.
        Specific to parallel nucleation, difference is that it considers sidedness + polarity.
        
        Parameters
        ----------
        mt_list : List of mt class
            MT list.
        pt : List of floats
            Pt of interest.
        t : Float
            Current time.
        
        Returns
        -------
        None.
        
        '''
        pos = arc_pos(pt, self.pol, self.seg, self.seg_dist)
        # pos = dist(pt, self.seg[0])
        # if pt[0] < self.seg[0][0]: #if x position behind, position is negative
        #     pos *= -1
        mt_in = [] #list of mts in the interval
        pol_in = [] #plus corresponding polarity + side info
        side_in = []
        for i in range(len(self.mts)): #examine when other mts hit this pt
            mt_idx = self.mts[i] #mt index
            polarity = self.rel_polarity[i]
            sidedness = self.mt_sides[i]
            # if mt_idx not in mt_none: #if there's an excluded mt
            mt = mt_list[mt_idx] #mt
            start = self.start_pos[i]
            special_case = False #special case where we're at a branch where there already exists a natural branch ahead but cannot resolve normally (mt starts exactly at the branch)
            #happens only when there are branch mts on both sides which creates a "crossing", but the crossing doesn't actually have
            #mts that traverse it, hence unable to resolve the mts' presence
            specialer = False #case where the pt is exactly at the mt tip. sorry, lots of cases
            if not mt.grow and mt.hit_bdry and (mt.seg[-1] not in [self.ext_pt, self.prev_pt]): #in the case where the pt is given
                current = None
                stall = mt.seg[-1] #last point on mt
                if stall in self.branch_pt: #reuse arc-pos for precision
                    pt_i = self.branch_pt.index(stall)
                    current = self.branch_pos[pt_i]
                else: #not a branch, due to pause state
                    if not no_bdl_id:#in this case, the branch point should not exist
                        assert mt.ext_mt == None
                    current = arc_pos(stall, self.pol, self.seg, self.seg_dist)
                tread_bool = True
                if self.rel_polarity[i] > 0 and (start < pos or special_case): #if growing in + dir of coord system
                    if mt.tread:
                        neg = start + (t-mt.tread_t)*mt.vt
                        if neg > pos:
                            tread_bool = False
                    if (pos < current or (pos==current and specialer)) and tread_bool: #if passed cross point
                        mt_in.append(mt.number)
                        pol_in.append(polarity)
                        side_in.append(sidedness)
                elif self.rel_polarity[i] < 0 and (start > pos or special_case):
                    if mt.tread:
                        neg = start - (t-mt.tread_t)*mt.vt
                        if neg < pos:
                            tread_bool = False
                    if (pos > current or (pos==current and specialer)) and tread_bool: #overtaken
                        mt_in.append(mt.number)
                        pol_in.append(polarity)
                        side_in.append(sidedness)
            else:
                dists = mt.seg_dist
                total_d = 0 #total dist grown
                tread_bool = True
                for d in range(len(dists)):
                    total_d += dists[d] #add segment distances
                if mt.grow: #if growing, add current tip length
                    total_d += t-mt.update_t[-1]
                elif not mt.grow and not mt.hit_bdry: #if shrinking, subtract
                    total_d -= v_s*(t-mt.update_t[-1])
                if self.rel_polarity[i] > 0 and (start < pos or special_case): #if growing in + dir of coord system
                    current = start + total_d
                    if mt.tread:
                        neg = start + (t-mt.tread_t)*mt.vt
                        if neg > pos:
                            tread_bool = False
                    if (pos < current or (pos==current and specialer)) and tread_bool: #if passed cross point
                        mt_in.append(mt.number)
                        pol_in.append(polarity)
                        side_in.append(sidedness)
                elif self.rel_polarity[i] < 0 and (start > pos or special_case):
                    current = start - total_d
                    if mt.tread:
                        neg = start - (t-mt.tread_t)*mt.vt
                        if neg < pos:
                            tread_bool = False
                    if (pos > current or (pos==current and specialer)) and tread_bool: #overtaken
                        mt_in.append(mt.number)
                        pol_in.append(polarity)
                        side_in.append(sidedness)
                    # td.append(total_d)
        assert mt2_n in mt_in
        return_side = None
        idx = None #find the mt and polarity
        if parallel:
            #I choose to consider all mts and randomly choose one, sidedness of new one +/-1 of chosen MT depending on distance to nucleation side
            compass = branch_geo(ref_th, meta_th, True) 
            idx = rnd.choice(range(0,len(side_in))) #we pick a random index
            if compass[1] == 'east':
                # idx = np.argmax(side_in)
                return_side = side_in[idx] + 1
            else:
                # idx = np.argmin(side_in)
                return_side = side_in[idx] - 1
            traj_polarity = pol_in[idx]*self.pol
            if traj_polarity < 0: #get mt angle
                if ref_th < pi:
                    ref_th += pi
                else:
                    ref_th -= pi
        else:
            #I make a coin flip on which outer MT to nucleate off of; easterly or westerly. Nucleation angles will be 
            idx = rnd.choice(range(0,len(side_in))) #we pick a random index
            return_side = side_in[idx]
            traj_polarity = pol_in[idx]*self.pol
            if traj_polarity < 0: #get mt angle
                if ref_th < pi:
                    ref_th += pi
                else:
                    ref_th -= pi
        # ref_th += branching_th
        # if ref_th > 2*pi: #must be under 2pi
        #     ref_th -= 2*pi
        # assert ref_th <= 2*pi
        return(ref_th, return_side)

def periodic(angle):
    #keeps angles within 2pi
    if angle > 2*pi: #must be under 2pi
        angle -= 2*pi
    elif angle < 0:
        angle += 2*pi
    return(angle)

def inter_traj(p1_eff,p2_eff,theta1,theta2,r_idx):
    '''
    Find intersection given pt and angles.

    Parameters
    ----------
    p1_eff : List of floats
        1st pt.
    p2_eff : List of floats
        2nd pt.
    theta1 : Float
        1st angle.
    theta2 : Float
        2nd angle.
    r_idx : Int
        Grid index, for whether intersection lies within grid.

    Returns
    -------
    Point of intersection or False if not.

    '''
    assert theta1 != pi/2 and theta2 != pi/2 and theta1 != 3*pi/2 and theta2 != 3*pi/2

    if abs(theta1 - theta2) < 1e-6 or abs(abs(theta1 - theta2) - pi) < 1e-10: #pi or abs(theta1 + theta2) == pi:
        return(False)
    else:
        m1,m2 = tan(theta1), tan(theta2) #slope intercept form
        if m1==m2:
            return(False)
        else:
            x1_eff,x2_eff,y1_eff,y2_eff = p1_eff[0],p2_eff[0],p1_eff[1],p2_eff[1] #effective pts are used for intersection calc
            b1,b2 = y1_eff-x1_eff*m1, y2_eff-x2_eff*m2
            x = (b2-b1)/(m1-m2) #equations for intersections
            y = m1*(b2-b1)/(m1-m2) + b1

            x_domain, y_domain = x_interval[r_idx], y_interval[r_idx] #find intervals
            xi, xf = x_domain[0],x_domain[1] #start, end of x domain
            yi, yf = y_domain[0],y_domain[1] #start, end of y domain
            if x > xi and x < xf and y > yi and y < yf:
                P = [x,y] #points and distances
                return(P)
            else:
                return(False)

def inter_bdry_traj(p, th, region): #calculate intersection of trajectories w/ bdry
    '''
    Calculate boundary intersection with a trajectory.

    Parameters
    ----------
    p : List of floats
        Pt of interest.
    th : Float
        Angle.
    region : Int
        Grid index.

    Returns
    -------
    Float. Distance intersection with boundary
    Array. Point of intersection
    String. Which wall it hits
    '''
    r_idx = region #region
    x_domain, y_domain = x_interval[r_idx], y_interval[r_idx] #find intervals
    xi, xf = x_domain[0],x_domain[1] #start, end of x domain
    yi, yf = y_domain[0],y_domain[1] #start, end of y domain
    x,y = p[0],p[1] #coordinates of points

    P = [0.,0.] #intersection point with bdry
    wall = None
    spec_case = False #bool for checking special case
    if x==xi and y==yi:
        spec_case = True
    if x==xi and y==yf:
        spec_case = True
    if x==xf and y==yi:
        spec_case = True
    if x==xf and y==yf:
        spec_case = True
    assert spec_case is False
    if x==xi: #starting on left boundary
        assert th<pi/2 or th>3*pi/2
        a1 = atan((yf-y)/(xf-xi))
        a4 = atan((xf-xi)/(y-yi)) + 3*pi/2
        if th<a1:
            P[0] = xf
            P[1] = tan(th)*(xf-x) + y
            wall = 'right'
        elif th<pi/2:
            P[1] = yf
            P[0] = (yf -y)/tan(th) + x
            wall = 'top'
        elif th<a4:
            P[1] = yi
            P[0] = (yi -y)/tan(th) + x
            wall = 'bottom'
        else:
            P[0] = xf
            P[1] = tan(th)*(xf-x) + y
            wall ='right'
    elif x==xf: #right bdry
        assert th>pi/2 and th<3*pi/2
        a2 = atan((xf-xi)/(yf-y)) + pi/2
        a3 = atan((y-yi)/(xf-xi)) + pi
        if th<a2:
            P[1] = yf
            P[0] = (yf -y)/tan(th) + x
            wall ='top'
        elif th<a3:
            P[0] = xi
            P[1] = tan(th)*(xi-x) + y
            wall ='left'
        else:
            P[1] = yi
            P[0] = (yi -y)/tan(th) + x
            wall ='bottom'
    elif y==yi: #bottom bdry
        assert th<pi
        a1 = atan((yf-yi)/(xf-x))
        a2 = atan((x-xi)/(yf-yi)) + pi/2
        # print(a1*180/pi)
        if th<a1:
            P[0] = xf
            P[1] = tan(th)*(xf-x) + y
            wall ='right'
        elif th<a2:
            P[1] = yf
            P[0] = (yf -y)/tan(th) + x
            wall ='top'
        else:
            P[0] = xi
            P[1] = tan(th)*(xi-x) + y
            wall = 'left'
    elif y==yf: #top bdry
        assert th>pi
        a3 = atan((yf-yi)/(x-xi)) + pi
        a4 = atan((xf-x)/(yf-yi)) + 3*pi/2
        if th<a3:
            P[0] = xi
            P[1] = tan(th)*(xi-x) + y
            wall = 'left'
        elif th<a4:
            P[1] = yi
            P[0] = (yi -y)/tan(th) + x
            wall ='bottom'
        else:
            P[0] = xf
            P[1] = tan(th)*(xf-x) + y
            wall ='right'
    else: #in between
        a1 = atan((yf-y)/(xf-x))
        a2 = atan((x-xi)/(yf-y)) + pi/2
        a3 = atan((y-yi)/(x-xi)) + pi
        a4 = atan((xf-x)/(y-yi)) + 3*pi/2
        assert th!=a1 and th!=a2 and th!=a3 and th!=a4
        if th < a1 or th>a4:
            P[0] = xf
            P[1] = tan(th)*(xf-x) + y
            wall = 'right'
        elif th < a2:
            P[1] = yf
            P[0] = (yf-y)/tan(th) + x
            wall = 'top'
        elif th < a3:
            P[0] = xi
            P[1] = tan(th)*(xi-x) + y
            wall = 'left'
        else:
            P[1] = yi
            P[0] = (yi-y)/tan(th) + x
            wall = 'bottom'
    return(P,wall)

class region_traj:
    '''
    Trajectories in a given region, along with all intersections w/ other traj. and bdry
    '''
    def __init__(self, mt_list, region):
        '''
        Initialize for a given region, using initial MTs.

        Parameters
        ----------
        mt_list : List of mt class
            MT class.
        region : Int
            Grid index.

        Returns
        -------
        None.

        '''
        self.region = region
        self.angle = [] #angles identified by its index in this list
        self.pt = [] #pt+angle uniquely defines the traj
        self.from_wall = [] #whether pt is on wall, if so which wall
        self.bdl_no = [] #corresponding bdl
        self.bdry_pt = [] #bdry collision given pt+angle NOTE: does not include backwards traj
        self.bdry_wall = [] #corresponding wall hit
        self.intersections = [] #triangular array for all intersection combos
        #add mt info
        region_list = [m for m in mt_list if m.region == region]
        k = 0 #traj number
        pseudo_idx = [] #indices of traj from pseudo bdry mts, need this for bdry info below
        pseudo_pt = [] #corresponding bdry pt
        for i in range(len(region_list)):
            mt1 = region_list[i]
            seg = mt1.seg
            angle = mt1.angle
            mt1.traj = [-1]*len(angle)
            for j in range(len(angle)):
                pt1, angle1 = seg[j], angle[j]
                if angle1 not in self.angle and pt1 not in self.pt: #avoid repeats
                    self.pt.append(pt1)
                    self.angle.append(angle1)
                    self.bdl_no.append(mt1.bdl)
                    mt1.traj[j] = k
                    if mt1.pseudo_bdry: #need to keep track of pseudo 
                        assert len(mt1.seg) == 2 #should only have seg on two walls
                        pseudo_idx.append(k)
                        pseudo_pt.append(mt1.seg[-1])
                    k+=1
                else:
                    mt1.traj[j] = self.angle.index(angle1)
        #calculate intersections
        I = len(self.pt)
        for i in range(I): #generate traj intersections between (i,j) for j<=i
            pts = [False]*(i+1) #intersection pts
            for j in range(i+1):
                if i != j:
                    pts[j] = inter_traj(self.pt[i],self.pt[j],self.angle[i],self.angle[j],region)
            self.intersections.append(pts)
            #bdry collisions
            if i not in pseudo_idx:
                bdry_res = inter_bdry_traj(self.pt[i],self.angle[i],self.region)
                self.bdry_pt.append(bdry_res[0])
                self.bdry_wall.append(bdry_res[1])
                self.from_wall.append(False)
            else: #in this case, already know wall info
                p_n = pseudo_idx.index(i)
                self.bdry_pt.append(pseudo_pt[p_n])
                self.bdry_wall.append('right')
                self.from_wall.append('left')
    def add_traj(self, pt1, angle1, mt, bdl_list, from_wall = False):
        '''
        Add new traj, calculate all intersections

        Parameters
        ----------
        pt1 : List of floats
            Pt.
        angle1 : Float
            Angle.
        mt : mt class
            MT of interest.
        bdl_list : List of bundle class
            bdl list.
        from_wall : Bool, optional
            Whether pt is on a wall. The default is False.

        Returns
        -------
        Int: traj ID

        '''
        if not no_bdl_id:
            if not (angle1 not in self.angle or pt1 not in self.pt):
                print(angle1 not in self.angle,pt1 not in self.pt)
                assert (angle1 not in self.angle or pt1 not in self.pt)
        self.pt.append(pt1)
        self.angle.append(angle1)
        if mt.bdl == None:
            if len(bdl_list) == 0:
                self.bdl_no.append(0)
            else:
                self.bdl_no.append(bdl_list[-1].number+1)
        else:
            self.bdl_no.append(mt.bdl)
        # self.mt_no.append(mt.number)
        bdry_res = inter_bdry_traj(pt1,angle1,self.region)
        self.bdry_pt.append(bdry_res[0])
        self.bdry_wall.append(bdry_res[1])
        if not from_wall:
            self.from_wall.append(from_wall)
        else: #append opposite wall
            wall = None
            if from_wall == 'top':
                wall = 'bottom'
            elif from_wall == 'bottom':
                wall = 'top'
            elif from_wall == 'left':
                wall = 'right'
            elif from_wall == 'right':
                wall = 'left'
            self.from_wall.append(wall)
        I = len(self.pt)
        pts = [False]*I
        for j in range(I):
            if j != I-1:
                pts[j] = inter_traj(self.pt[-1],self.pt[j],self.angle[-1],self.angle[j],self.region)
        self.intersections.append(pts)
        return(I-1)
    def add_all(self, bdl_n1, pt1, angle1, bdry1, wall1, from_wall1):
        '''
        When created from a purge, append all entries that are to be kept

        Parameters
        ----------
        bdl_n1 : Int
            bundle ID.
        pt1 : List of floats
            pt.
        angle1 : Float
            angle.
        bdry1 : List of floats
            Bdry pt.
        wall1 : String
            Which wall.
        from_wall1 : Bool
            Whether the pt is on the wall.

        Returns
        -------
        None.

        '''
        self.bdl_no.append(bdl_n1)
        self.pt.append(pt1)
        self.angle.append(angle1)
        self.bdry_pt.append(bdry1)
        self.bdry_wall.append(wall1)
        self.from_wall.append(from_wall1)
    def check_traj_meta(self, pt_meta, angle_meta):
        '''
        Method of traj object
        New meta traj, calculate all intersections WITHOUT modifying existing
        intersection data

        Parameters
        ----------
        pt1 : List of floats
            Pt.
        angle1 : Float
            Angle.
        mt : mt class
            MT of interest.
        bdl_list : List of bundle class
            bdl list.
        from_wall : Bool, optional
            Whether pt is on a wall. The default is False.

        Returns
        -------
        Int: traj ID

        '''
        
        # bdry_res = inter_bdry_traj(pt1,angle1,self.region)
        # self.bdry_pt.append(bdry_res[0])
        # self.bdry_wall.append(bdry_res[1])
        
        I = len(self.pt)+1
        pts = [False]*I #used to find MT intersections
        for j in range(I):
            if j != I-1:
                pts[j] = inter_traj(pt_meta,self.pt[j],angle_meta,self.angle[j],self.region)
        return(pts, I-1)

def inter_r_bdry2(MT, MT_list, bdl_list, region_list, free=False, to_wall = False):
    '''
    Retrieve bdry calculation and see if it deflects first.

    Parameters
    ----------
    MT : mt class
        MT of interest.
    MT_list : List of mt class
        MT list.
    bdl_list : List of bundle class
        Bdl list.
    region_list : List of region_traj
        Traj list.
    free : Bool, optional
        Whether MT can bend. The default is False.
    to_wall : Bool, optional
        Whether MT is going up a bdl. The default is False.

    Returns
    -------
    Float. Distance intersection with boundary
    Array. Point of intersection
    String. Which wall it hits

    '''
    p, th = MT.seg[-1], MT.angle[-1]
    traj_n = MT.traj[-1]
    r_idx = MT.region #region
    region = region_list[r_idx]
    x_domain, y_domain = x_interval[r_idx], y_interval[r_idx] #find intervals
    xi, xf = x_domain[0],x_domain[1] #start, end of x domain
    yi, yf = y_domain[0],y_domain[1] #start, end of y domain
    # x,y = p[0],p[1] #coordinates of points
    P, wall = None, None
    if to_wall: #mt traveling up to original wall pt
        assert region.from_wall[traj_n] != False
        P1 = region.pt[traj_n]
        P = [P1[0],P1[1]] #be careful: do not mess with the reference
        wall = region.from_wall[traj_n]
        # assert wall in ['top','bottom','left','right']
    else:
        if abs(region.angle[traj_n] - th) > 1e-13: #not always equal due to +pi then - pi
            print('ERROR',traj_n,region.angle[traj_n] - th, traj_n,region.angle[traj_n],th)
            assert abs(region.angle[traj_n] - th) < 1e-13 #should not be calculating bdry collisions backwards
        P1 = region.bdry_pt[traj_n] #intersection point with bdry
        P = [P1[0],P1[1]]
        wall = region.bdry_wall[traj_n]
    D = dist(p,P)
    if free:
        D1 = 0 #consider if it deflects
        if not MT.from_bdry or len(MT.seg)>1: #if it's not from bdry or more than 1 segment
            D1 = MT.tip_l
        elif MT.from_bdry: #from bdry
            D1 = MT.tip_l - (MT.update_t[-1] - MT.prev_t)
            if D1<0:
                print(MT.number,MT.update_t[-1],MT.prev_t)
            assert MT.update_t[-1] - MT.prev_t < MT.tip_l
        if D1<D: #if deflects before intersection w/ bdry
            D = D1 #reassign
            wall = 'deflect' #keyword
            P[0] = p[0] + D1*cos(th) #deflected point
            P[1] = p[1] + D1*sin(th)
    if abs(P[0])>xdomain[1] or abs(P[1])>ydomain[1]:
        print('MT has gone crazy: ', MT.number)
        print('Point: ', P)
        print(wall, traj_n)
        assert abs(P[0])<=xdomain[1] and abs(P[1])<=ydomain[1]
    return(D,P,wall)

def inter2(p1,p2,theta1,theta2,traj1_n,traj2_n,r_idx,region_list):
    '''
    Retrieve intersection of trajs.

    Parameters
    ----------
    p1 : List of floats
        1st pt.
    p2 : List of floats
        2nd pt.
    theta1 : Float
        1st angle.
    theta2 : Float
        2nd angle.
    traj1_n : Int
        1st traj ID.
    traj2_n : Int
        2nd traj ID.
    r_idx : Int
        Grid index.
    region_list : List of region_traj
        Traj list.

    Returns
    -------
    Bool. Whether it collides.
    List of flaots. Pt of intersection.
    List of floats. Distances from pt1 and pt2.

    '''
    region = region_list[r_idx]
    if traj1_n < traj2_n:
        traj1_n, traj2_n = traj2_n, traj1_n #pair must be reverse-ordered
    if traj1_n > len(region.intersections)-1 or traj2_n > len(region.intersections[traj1_n]):
        print(traj1_n,len(region.intersections))
    P = region.intersections[traj1_n][traj2_n]
    if P == False:
        return(False, None, None)
    else:
        x1,x2 = p1[0],p2[0]
        y1,y2 = p1[1],p2[1]
        x,y = P[0], P[1]
        sign_x1, sign_x2 = 1, 1 #determine whether new pt in the dir of both mts (not behind)
        sign_y1, sign_y2 = 1, 1
        if theta1 > pi/2 and theta1 < 3*pi/2:
            sign_x1 = -1
        if theta2 > pi/2 and theta2 < 3*pi/2:
            sign_x2 = -1
        if theta1 > pi:
            sign_y1 = -1
        if theta2 > pi:
            sign_y2 = -1
        # if np.sign(x-x1)==np.sign(cos(theta1)) and np.sign(x-x2)==np.sign(cos(theta2)): #if in trajectory
        if np.sign(x-x1)==sign_x1 and np.sign(x-x2)==sign_x2 and np.sign(y-y1)==sign_y1 and np.sign(y-y2)==sign_y2: #if in trajectory
            D = [dist(P,p1),dist(P,p2)]
            return(True, P, D)
        else:
            return(False, None, None)

def bdl_exist(bdl, bdl_list):
    remain = False #whether there are mts remaining on ANY ext/prev bdl
    if len(bdl.mts) != 0: #not dead
        remain = True
    else: #no mts on bdl anymore, can erase if ext/prev also don't have any
        temp_bdl = bdl
        ext_id = bdl.ext_bdl
        prev_id = bdl.prev_bdl
        while (ext_id is not None) and (not remain):
            temp_bdl = bdl_list[ext_id] #examine extension
            ext_id = temp_bdl.ext_bdl
            if len(temp_bdl.mts) != 0:
                remain = True
        temp_bdl = bdl #reassign for searching prev bdls
        while (prev_id is not None) and (not remain):
            temp_bdl = bdl_list[prev_id] #examine extension
            prev_id = temp_bdl.prev_bdl
            if len(temp_bdl.mts) != 0:
                remain = True
    return(remain)

def earliest_event(mt, t, mt_list, bdl_list, event_list):
    assert mt.grow
    # easy_list = ['1hit2','cross_bdl','cross_br','1catch2','1catch2_m','deflect','follow_bdl' \
    #              'top','bottom', 'left','right'] #we just need to compare mt1_n for these
    # stoch_list = ['sp_catastrophe', 'grow_to_pause', 'nucleate']
    # root_mt = mt
    # while root_mt.prev_mt is not None:
    #     new_no = root_mt.prev_mt
    #     root_mt = mt_list[new_no]
    # mt_event_list = [e for e in event_list if e.t != t]
    comparison_events = []
    for i in range(len(event_list)): #assemble event list into 1D array rather than N-D where N is number of regions
        comparison_events += event_list[i]
    comparison_events.sort(key=lambda x: x.t)
    later_t = [e.t for e in comparison_events if e.t > t]
    earlier_t = [e.t for e in comparison_events if e.t <= t] #all events that are equal or earlier than this
    del earlier_t[0] #first element will be current event
    t2 = comparison_events[1].t #earliest time
    # t2 = event_list[1].t
    T = None #next event time
    if len(earlier_t) >= 1:
        warnings.warn('Earlist event times found {}, current time is {}'.format(earlier_t,t))
        print(comparison_events[1].pt)
        # assert t2 > t
        t2 = later_t[0]
    diff = t2-t
    if diff < tub:
        T = t+diff/2
        if diff/2<1e-12:
            warnings.warn('MT stepback to small length: '+str(diff/2))
            print(comparison_events[1].pt, comparison_events[1].t)
    else:
        T = t+tub
    new_dist = T - mt.update_t[-1] #calculate
    th = mt.angle[-1]
    old_pt = mt.seg[-1]
    x, y = old_pt[0] + new_dist*cos(th), old_pt[1] + new_dist*sin(th)
    new_pt = [x,y]
    return(new_pt, T)

def cat_rate(angle,t):
    assert angle <= pi/2
    t_real = t*conv/60/60
    if angle_dep_cat:# and t_real <= 5:
        return(r_c*(1+ep*sin(angle))/(1+2*ep/pi))
    else:
        return(r_c)

#functions for LDD nucleation
from parameters import R_meta, r_u, D_meta, accept_MT, accept_unbound

def meta_traj_angle(N_traj=6):
    #generate angle for meta traj
    j = rnd.randrange(0,N_traj-1,1)
    th = j*2*pi/N_traj
    shift = rnd.uniform(0,2*pi)
    th += shift
    if th >= 2*pi:
        th -= 2*pi
    return(th)

def inter_meta(p1,p2,theta1,theta2,traj1_n,traj2_n,r_idx,intersections):
    '''
    Retrieve intersection of trajs. Only for Meta traj.
    Difference from inter2: intersections are passed into this, rather than via
    traj object.

    Parameters
    ----------
    p1 : List of floats
        1st pt.
    p2 : List of floats
        2nd pt.
    theta1 : Float
        1st angle.
    theta2 : Float
        2nd angle.
    traj1_n : Int
        1st traj ID.
    traj2_n : Int
        2nd traj ID.
    r_idx : Int
        Grid index.
    region_list : List of region_traj
        Traj list.

    Returns
    -------
    Bool. Whether it collides.
    List of flaots. Pt of intersection.
    List of floats. Distances from pt1 and pt2.

    '''
    if traj1_n < traj2_n:
        traj1_n, traj2_n = traj2_n, traj1_n #pair must be reverse-ordered
    if traj1_n > len(intersections)-1 or traj2_n > len(intersections):
        print(traj1_n,len(intersections))
    P = intersections[traj2_n]
    if P == False:
        return(False, None, None)
    else:
        x1,x2 = p1[0],p2[0]
        y1,y2 = p1[1],p2[1]
        x,y = P[0], P[1]
        sign_x1, sign_x2 = 1, 1 #determine whether new pt in the dir of both mts (not behind)
        sign_y1, sign_y2 = 1, 1
        if theta1 > pi/2 and theta1 < 3*pi/2:
            sign_x1 = -1
        if theta2 > pi/2 and theta2 < 3*pi/2:
            sign_x2 = -1
        if theta1 > pi:
            sign_y1 = -1
        if theta2 > pi:
            sign_y2 = -1
        # if np.sign(x-x1)==np.sign(cos(theta1)) and np.sign(x-x2)==np.sign(cos(theta2)): #if in trajectory
        if np.sign(x-x1)==sign_x1 and np.sign(x-x2)==sign_x2 and np.sign(y-y1)==sign_y1 and np.sign(y-y2)==sign_y2: #if in trajectory
            D = [dist(P,p1),dist(P,p2)]
            return(True, P, D)
        else:
            return(False, None, None)

def compare_meta(pt0, th, mt2, t, intersections):
    #compare_return but specific to comparing meta trajectory w/ trajectories
    output = compare_return() #initiate output
    tol = 1e-13 #numerical tolerance for intersection distance, otherwise no intersection
    r_idx = mt2.region
    traj_meta_n = len(intersections)-1 #meta traj number, doesn't physically exist
    if not mt2.exist:
        output.dist = None
        output.policy = 'no_collision'
        output.point = None
        return(output)
    elif (mt2.grow): #MT2 growing
        seg2 = mt2.seg #assign segment points
        traj2 = mt2.traj
        th2 = mt2.angle
        old_dist2 = mt2.seg_dist
        l2 = len(seg2) #for indexing
        p2_prev = seg2[-1] #last updated point and time
        t_prev2 = mt2.update_t[-1]

        col_dist1t2 = [] #collision distances from 1 to 2

        point_1t2 = [] #store their respective collision locations

        seg2_idx = [] #store indices of collision seg_dist
        #We now check if mt1 collides w/ any segments of mt2
        for i in range(l2):
            p2 = seg2[i] #point traj to be collided with
            col_result = inter_meta(pt0, p2, th, th2[i], traj_meta_n, traj2[i], r_idx, intersections)
            if col_result[0] is True:
                d1, d2 = col_result[2][0], col_result[2][1] #distance to collision from mt1 and mt2
                pt = col_result[1] #point of collision
                if i==l2-1: #checking collision w/ the two growing ends
                    d_g = t-t_prev2 #distance grown by mt's is always max
                    if d_g>d2: #mt2 needs to exist there
                        store = True
                        if mt2.tread:# and i==seg_start2 and d2<tr_pos2:
                            d_tr2 = (t-mt2.tread_t)*mt2.vt #amount treaded
                            tread_res = tread_dist(d_tr2, old_dist2, l2)
                            tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                            if (seg_start2 > i) or (seg_start2 == i and d2 < tr_pos2):
                                store = False #treadmilled past the pt
                        if store:
                            col_dist1t2.append(d1)
                            point_1t2.append(pt)
                            seg2_idx.append(i)
                else:#check intersection of mt1 head w/ previous mt2 segments
                    if d2<= old_dist2[i]: #must be less than segment length for collision
                        store = True
                        if mt2.tread:# and i==seg_start2 and d2<tr_pos2:
                            d_tr2 = (t-mt2.tread_t)*mt2.vt #amount treaded
                            tread_res = tread_dist(d_tr2, old_dist2, l2)
                            tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                            if (seg_start2 > i) or (seg_start2 == i and d2 < tr_pos2):
                                store = False #treadmilled past the pt
                        if store:
                            col_dist1t2.append(d1) #store collision distance
                            point_1t2.append(pt)
                            seg2_idx.append(i)
        which_min, i_1t2, = None, None #declare
        if len(col_dist1t2) ==0:
            output.policy = 'no_collision'
            return(output)
        else:
            which_min = 0
            i_1t2 = np.argmin(col_dist1t2)
            if col_dist1t2[i_1t2]<=0: #TODO this should never happen, what's going on here?
                output.policy = 'no_collision'
            else:
                output.policy = '1hit2'
                output.point = point_1t2[i_1t2]
                output.dist = col_dist1t2[i_1t2]
                output.idx = seg2_idx[i_1t2]
    elif not mt2.hit_bdry: #IF ONE END IS SHRINKING
        assert mt2.grow is False
        which = None #which tip hits
        mts = mt2
        which = 1
        assert len(mts.seg) >= 2
        seg2 = mts.seg #assign segment points
        traj2 = mts.traj
        th2 = mts.angle
        old_dist2 = mts.seg_dist
        l2 = len(seg2) #for indexing
        t_prev2, p2_prev = mts.update_t[-1], seg2[l2-1] #last updated point and time
        # p2_prev = seg2[-1] #last updated point and time
        assert(len(old_dist2) == len(seg2)-1)

        mt2_l = np.sum(old_dist2)#total length of shrinking mt, cannot got lower than this
        col_dist1t2 = [mt2_l/v_s - (t-t_prev2)] #collision distances from 1 to 2, growing can only collide w/ shrinking
        point_1t2 = [[0,0]] #placeholder corresponding to total shrinkage

        seg2_idx = [0]
        #We now check if mt1 collides w/ any segments of mt2
        for i in range(l2-1): #only up to second last pt matters when shriking
            if i==l2-2: #checking collision w/ the two dynamic ends
                p2 = seg2[i] #point traj to be collided with
                col_result = inter_meta(pt0, p2, th, th2[i], traj_meta_n, traj2[i], r_idx, intersections)
                if col_result[0] is True:
                    d1, d2 = col_result[2][0], col_result[2][1] #distance to collision from mt1 and mt2
                    if d2<= old_dist2[i]: #has to collide on segment
                        # d_g = d1 - (t-t_prev1) #distance of mt1 collision, also time taken to grow this
                        d_s = v_s*(t-t_prev2) #distance shrank
                        d_segf = old_dist2[i] - d_s #total distance left on the segment
                        assert d2 != d_segf
                        if d2 < d_segf:
                            store = True
                            if mts.tread:# and i==seg_start2 and d2<tr_pos2:
                                d_tr2 = (t-mts.tread_t)*mts.vt #amount treaded
                                tread_res = tread_dist(d_tr2, old_dist2, l2)
                                tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                                if (seg_start2 > i) or (seg_start2 == i and d2 < tr_pos2):
                                    store = False #treadmilled past the pt
                            if store:
                                col_dist1t2.append(d1)
                                point_1t2.append(col_result[1])
                                seg2_idx.append(i)
            else:#check intersection of mt1 head w/ previous mt2 segments
                p2 = seg2[i] #point traj to be collided with
                col_result = inter_meta(pt0, p2, th, th2[i], traj_meta_n, traj2[i], r_idx, intersections)
                if col_result[0] is True:
                    d1, d2 = col_result[2][0], col_result[2][1] #distance to collision from mt1 and mt2
                    if d2<= old_dist2[i]: #has to collide on segment
                        # d_g = d1 - (t-t_prev1) #distance/time taken to grow starting at t
                        d_s = v_s*(t-t_prev2) #distance shrank of mt2
                        d_segf = mt2_l - d_s #length of mt2 left
                        for k in range(i): #total length of mt2 at intersection point, add prev segs
                            d2 += old_dist2[k]
                        assert d2 != d_segf
                        if d2< d_segf: #if shrank to less than intersection distance
                            store = True
                            if mts.tread:# and i==seg_start2 and d2<tr_pos2:
                                 d_tr2 = (t-mts.tread_t)*mts.vt #amount treaded
                                 tread_res = tread_dist(d_tr2, old_dist2, l2)
                                 tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                                 if (seg_start2 > i) or (seg_start2 == i and col_result[2][1] < tr_pos2): #d2 was edited, use original value
                                     store = False #treadmilled past the pt
                            if store:
                                col_dist1t2.append(d1)
                                point_1t2.append(col_result[1])
                                seg2_idx.append(i)
        i_1t2 = None #declare
        if len(col_dist1t2)==1: #if not additional points are added, mt shrinks away before collision
            output.dist = col_dist1t2[0]
            if which == 1:
                output.policy = 'no_collision'#'2disap'
            else:
                output.policy = 'no_collision'#'1disap'
        else: #collision occurs
            i_1t2 =  np.argmin(col_dist1t2)
            min_1t2 = col_dist1t2[i_1t2]
            output.point = point_1t2[i_1t2]
            output.dist = col_dist1t2[i_1t2]
            output.idx = seg2_idx[i_1t2]
            if which == 2:
                output.policy = '2hit1'
            else:
                output.policy = '1hit2'
    elif mt2.hit_bdry: #IF ONE IS GROWING AND ONE IS ON THE BORDER
        which = None
        mtb = mt2
        assert len(mtb.seg) >= 2
        seg2 = mtb.seg #assign segment points
        traj2 = mtb.traj
        th2 = mtb.angle
        old_dist2 = mtb.seg_dist
        l2 = len(seg2) #for indexing
        t_prev2, p2_prev = mtb.update_t[-1], seg2[l2-1] #last updated point and time
        p2_prev = seg2[-1] #last updated point and time
        # assert mt1.checkd() and mt2.checkd()
        col_dist1t2 = [] #collision distances from 1 to 2
        point_1t2 = [] #store their respective collision locations
        neg_dist = 0

        seg2_idx=[]
        #We now check if mt1 collides w/ any segments of mt2
        for i in range(l2-1):
            p2 = seg2[i] #point traj to be collided with
            col_result = inter_meta(pt0, p2, th, th2[i], traj_meta_n, traj2[i], r_idx, intersections)
            if col_result[0] is True:
                d1, d2 = col_result[2][0], col_result[2][1] #distance to collision from mt1 and mt2
                pt = col_result[1]
                if d2<= old_dist2[i]: #must be less than segment length for collision
                    # d_g = (t-t_prev1) #total distance grown since t
                    # if d_g<= tol:
                    #     neg_dist += 1
                    # else:
                    store = True
                    if mtb.tread:# and i==seg_start2 and d2<tr_pos2:
                        d_tr2 = (t-mtb.tread_t)*mtb.vt #amount treaded
                        tread_res = tread_dist(d_tr2, old_dist2, l2)
                        tr_pos2, seg_start2 = tread_res[0], tread_res[1]
                        if (seg_start2 > i) or (seg_start2 == i and d2 < tr_pos2):
                            store = False #treadmilled past the pt
                        # store = False #treadmilled past the pt
                    if store:
                        col_dist1t2.append(d1) #store collision distance
                        point_1t2.append(pt)
                        seg2_idx.append(i)
        if len(col_dist1t2)==0 or neg_dist>0:
            output.policy = 'no_collision'
        else:
            i_1t2 = np.argmin(col_dist1t2)
            min_1t2 = col_dist1t2[i_1t2]
            output.point = point_1t2[i_1t2]
            output.dist = col_dist1t2[i_1t2]
            output.idx = seg2_idx[i_1t2]
            if which == 2:
                output.policy = '2hit1'
            else:
                output.policy = '1hit2'
    if output.policy != 'no_collision':
        assert output.dist > tol
        x = x_interval[mt2.region] #discard events outside of region
        y = y_interval[mt2.region]
        xi, xf = x[0], x[1]
        yi, yf = y[0], y[1]
        if output.point[0] >= xf or output.point[0] <= xi or output.point[1] >= yf or output.point[1] <= yi:
            output.policy = 'no_collision'
    return(output)

def find_meta_col(pt, th, mt_list, t, intersections):
    #determine all collisions w/ meta traj
    L = len(mt_list)
    col_list = [] #list of collision results and mt numbers
    for i in range(L): #find whether meta traj collides w/ mts in region
        mt2 = mt_list[i]
        col_res = compare_meta(pt, th, mt2, t, intersections)
        if col_res.policy != 'no_collision':
            col_list.append([col_res, mt2.number])
    if len(col_list) != 0:
        col_list.sort(key=lambda x: x[0].dist)
    return(col_list)

def nuc_flip(bound=True):
    #for figuring if MT/ubound nucleation occurs after chosen
    dissociates = False
    #assumes we already rescales s.t. bound nuc has 100% probability
    if not bound:
        X = rnd.uniform(0,1)
        if X >= accept_unbound:
            dissociates = True
    # dissociates = False
    return(dissociates)

def identify_bdry_pt(pt, wall, old_region):
    #get bdry pt and region
    px, py = pt[0], pt[1]
    x1, x2 = xdomain[0], xdomain[1]
    y1, y2 = ydomain[0], ydomain[1]
    if px == x1:
        px = x2
    elif px == x2:
        px = x1
    elif py == y1:
        py = y2
    elif py == y2:
        py = y1
    pt_new = [px, py]
    new_region = old_region
    #assign new region
    if wall == 'top':
        new_region = (old_region + grid_w)%(grid_w*grid_l)
    elif wall == 'bottom':
        new_region = (old_region - grid_w)%(grid_w*grid_l)
    elif wall =='left':
        mod = np.floor(old_region/grid_w)
        horiz = old_region - grid_w*mod
        horiz = (horiz-1)%grid_w
        new_region = int(grid_w*mod + horiz)
    elif wall =='right':
        mod = np.floor(old_region/grid_w)
        horiz = old_region - grid_w*mod
        horiz = (horiz+1)%grid_w
        new_region = int(grid_w*mod + horiz)
    return(pt_new, new_region)

def nucleate(traj_list, mt_list, pt, t):
    #need to generate below randomly
    original_pt = pt
    dissociates = True
    unbd_nuc = False #whether results in iso nuc
    mt2_n = None
    traj = None
    d = 0 #distance from start to nucleation pt
    nuc_pt = [0,0] #nucleation pt
    ins_pt = pt #insertion pt
    th = meta_traj_angle() #pick meta traj
    K = rnd.uniform(0, 1)
    unbd_nuc_t = -log(K)/r_u #sample distance travelled til unbound nuc
    unbd_nuc_d = sqrt(4*D_meta*unbd_nuc_t) #convert to distance
    checking = True #iterate until no need to check
    r = which_region(ins_pt) #get region for traj calc. Recalculated within loop if needed
    while checking:
        assert d <= R_meta and d <= unbd_nuc_d
        traj_r = traj_list[r]
        mt_r = [m for m in mt_list if m.exist and m.region==r] #mts to compare with
        traj_res = traj_r.check_traj_meta(ins_pt,th) #intersection info
        intersections = traj_res[0]
        # print(intersections)
        col_pt = find_meta_col(ins_pt, th, mt_r, t, intersections) #get traj intersections w/ metatraj
        # print(th*180/pi,len(col_pt),r,unbd_nuc_d, ins_pt)
        if len(col_pt) == 0: #no collision w/ mts, check bdry collision
            assert d <= R_meta and d <= unbd_nuc_d
            bdry_res = inter_bdry_traj(ins_pt, th, r) #find bdry pt
            wall, bdry_pt = bdry_res[1], bdry_res[0]
            d_gridtobdry = dist(ins_pt,bdry_pt) #distance from grid OR original pt to bdry
            d_tobdry = d+d_gridtobdry #total distance to bdry, may be equal to d_gridtobdry
            if d_tobdry <= R_meta: #all within meta traj scope
                if d_tobdry > unbd_nuc_d: #nucleates before reaches bdry
                    # print('1')
                    d_fromgrid = unbd_nuc_d - d #dist from grid to unbd nuc pt
                    assert d_fromgrid > 0
                    nuc_pt[1] = ins_pt[1]+d_fromgrid*sin(th)
                    nuc_pt[0] = ins_pt[0]+d_fromgrid*cos(th)
                    dissociates = nuc_flip(bound=False) #whether dissociates
                    unbd_nuc = True
                    checking = False
                else: #on to a new region
                    # print('2')
                    new_stuff = identify_bdry_pt(bdry_pt, wall, r) #identify edges if needed
                    ins_pt, r = new_stuff[0], new_stuff[1]
                    d = d_tobdry
                    checking = True
            else: #outside scope of meta traj
                unbd_nuc_d = min(unbd_nuc_d,R_meta) #either nucleates on the edge or interior of scope
                d_fromgrid = unbd_nuc_d-d #distance from current pt
                assert d_fromgrid > 0
                nuc_pt[1] = ins_pt[1]+d_fromgrid*sin(th)
                nuc_pt[0] = ins_pt[0]+d_fromgrid*cos(th)
                dissociates = nuc_flip(bound=False) #whether dissociates
                unbd_nuc = True
                checking = False
        else:
            checking = False #do not to iterate in grids anymore, nuc/dissociation must happen
            collision = col_pt[0][0]
            mt2_n = col_pt[0][1]
            pt = collision.point
            # print('nuc colision', pt, mt2_n, collision.idx)
            traj = mt_list[mt2_n].traj[collision.idx]
            d_fromgrid = collision.dist #distance from grid OR original pt to new collision pt
            d_tomt = d + d_fromgrid
            if d_tomt <= R_meta:
                if d_tomt > unbd_nuc_d: #unbd nucleation
                    d_fromgrid = unbd_nuc_d - d
                    assert d_fromgrid >= 0
                    nuc_pt[1] = ins_pt[1]+d_fromgrid*sin(th)
                    nuc_pt[0] = ins_pt[0]+d_fromgrid*cos(th)
                    dissociates = nuc_flip(bound=False)#or true depending on flip
                    unbd_nuc = True
                else: #bd'd nucleation
                    nuc_pt = pt
                    dissociates = nuc_flip()
                    unbd_nuc = False
            else: #outside scope of meta traj
                unbd_nuc_d = min(unbd_nuc_d,R_meta) #either nucleates on the edge or interior of scope
                d_fromgrid = unbd_nuc_d-d #distance from current pt
                assert d_fromgrid > 0
                nuc_pt[1] = ins_pt[1]+d_fromgrid*sin(th)
                nuc_pt[0] = ins_pt[0]+d_fromgrid*cos(th)
                dissociates = nuc_flip(bound=False)
                unbd_nuc = True
    if not dissociates:
        check_nuc_dist(original_pt,nuc_pt, th) #double check their distances
    return(nuc_pt, dissociates, unbd_nuc, mt2_n, traj, th)

def check_nuc_dist(pt1,pt2,th):
    #checks that nucleation distance isn't violated
    #pt1 original, pt2 new. order matters
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    pt3 = [x2,y2]
    if th<= pi/2 or th >= 3*pi/2:
        if x2 < x1:
            pt3[0] = x2+xdomain[1]
    else:
        if x2 > x1:
            pt3[0] = x2-xdomain[1]
    if th <= pi:
        if y2 < y1:
            pt3[1] = y2+ydomain[1]
    else:
        if y2 > y1:
            pt3[1] = y2-ydomain[1]
    nuc_dist = dist(pt3,pt1)
    if nuc_dist > R_meta:
        epsilon = abs(R_meta - nuc_dist)
        assert epsilon < 1e-10

def determine_branch_compass(br_th, mt_angle, ref_th):
    #figure out whether ref bundle is in/out branch of nucleated mt 
    #mt_angle: angle of the mt that the complex is branching off of
    #ref_th: angle of parent mt i.e. representative mt
    #br_th: angle of nucleation w.r.t. mt_angle
    switch = False
    if abs(mt_angle-ref_th) > 1e-12:
        switch = True #parent mt not same angle as mt angle
    inward = True
    if br_th >= pi/2 and br_th <= 3*pi/2:
        inward  = False
    # if switch:
    #     if inward:
    #         inward = False
    #     else:
    #         inward = True
    return(inward)

def determine_branch_compass2(br_th, mt_angle, ref_th):
    #figure out whether ref bundle is in/out branch of nucleated mt 
    #mt_angle: angle of the mt that the complex is branching off of
    #ref_th: angle of parent mt i.e. representative mt
    #br_th: angle of nucleation w.r.t. mt_angle
    switch = False
    if abs(mt_angle-ref_th) > 1e-12:
        switch = True #parent mt not same angle as mt angle
    inward = True
    if br_th >= pi/2 and br_th <= 3*pi/2:
        inward  = False
    if switch:
        if inward:
            inward = False
        else:
            inward = True
    return(inward)

#load in lookup table for sampling ellipse distr
from inverse_transform import make_ellipse_table
make_ellipse_table()
with open('ellipse_table.pickle', 'rb') as pick_in:
    lookup_table = pickle.load(pick_in)

dy_lookup = lookup_table['dtheta']
theta_inverse = lookup_table['theta_inverse']

def sample_angle():
    #generates elliptical angle for branched nucleation
    f_forward = 0.31
    f_backward = 0.07
    Y = rnd.uniform(0,1) #for choosing which mode of nucleation
    # Y = rnd.uniform(0,1) #TODO ERASE
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

def ghost_pt(original_pt, angle):
    #generate a step-back pt for branched nucleation bundles, ghost pt
    x0, y0 = original_pt[0], original_pt[1]
    d = tub*2
    r0 = which_region(original_pt)
    r_new = 0.5
    pt_new = None
    while r0 != r_new:
        d = d/2
        x1 = x0 - d*cos(angle)
        y1 = y0 - d*sin(angle)
        pt_new = [x1,y1]
        r_new = which_region(pt_new)
    return(pt_new, d)

def add_pi(angle, added_angle):
    #adding pi while minimizing floating point errors
    if added_angle != 0:
        if angle >= pi:
            angle -= pi
        else:
            angle += pi
    return(angle)

def step_forward(mt1, original_pt):
    #same idea as ghost_pt but step forward to avoid weird region stuff happening with bundle traffic
    r_true = mt1.region
    r_new = which_region(original_pt)
    x0, y0 = original_pt[0], original_pt[1]
    d = tub*2
    pt_new = None
    angle = mt1.angle[-1]
    while r_true != r_new:
        d = d/2
        x1 = x0 + d*cos(angle)
        y1 = y0 + d*sin(angle)
        pt_new = [x1,y1]
        r_new = which_region(pt_new)
    return(pt_new)

if __name__ == '__main__':
    p1 = [2,3]
    p2 = [0,0]
    M1 = mt(1)
    M1.seg = [[0.2,0.1]]
    M1.angle = [pi]
    bdl_list = []
    B = bundle(M1,bdl_list)

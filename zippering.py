#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:09:10 2021

@author: tim
"""
import numpy as np
from math import sin, cos, pi
from comparison_fns import dist
import sys
from parameters import no_bdl_id, dr
# import random as rnd
d =  dr#5*1e-3/16#2.5e-3 #distance away from MT for zippering
dr_tol = d/sin(.01) #tolerance for step-back of branching mt

def zip_cat(angle1,angle2,pt,pt_prev,r):
    '''
    Determine whether the intersection results in catastrophe

    Parameters
    ----------
    angle1 : angle of tip which collides
    angle2 : angle of barrier MT
    pt_prev : previous vertex of incoming MT
    pt : point of intersection
    r: 0 or 1 random number
    Returns
    -------
    new_angle: entrainment angle, if not catastrophe/crossover
    new_pt: starting pt of entrained MT segment
    resolve: cross, zipper+/-, catas
    col_pt: end point of incoming MT

    '''
    resolve = 'cross'
    th2,th1 = max(angle1,angle2),min(angle1,angle2)
    # print(angle1/pi,angle2/pi)
    th_crit = 2*pi/9 #critical angle
    #dr = None#declare
    new_pt = [pt[0],pt[1]]
    col_pt = [pt[0],pt[1]]#None #collision pt also steps back a bit
    new_angle = angle1
    if th2 > 3*pi/2 and th1<pi/2:
        a1 = 2*pi-th2 #one angle
        a2 = th1
        b = a1+a2 #incident angle
        if th2 == angle1: #incoming angle is largest
            if b >= pi/2: #incident angle is large
                b2 = pi - b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = pi + th1
                else: #catastrophe TODO
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
            else: #incident angle is good
                b2 = b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th1
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
        else:
            if b >= pi/2: #incident angle is large
                b2 = pi - b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th2-pi
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
            else: #incident angle is good
                b2 = b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th2
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
    elif th2 > pi/2 and th2< pi and th1<pi/2:
        a1 = pi-th2 #one angle
        a2 = th1
        b = a1+a2 #incident angle
        if th2 == angle1: #incoming angle is largest
            if b >= pi/2: #incident angle is large
                b2 = pi - b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th1
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
            else: #incident angle is good
                b2 = b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th1+pi
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
        else:
            if b >= pi/2: #incident angle is large
                b2 = pi - b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th2
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
            else: #incident angle is good
                b2 = b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th2+pi
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
    elif th2 > 3*pi/2 and th1<3*pi/2 and th1 > pi:
        a1 = th1- pi#one angle
        a2 = 2*pi - th2
        b = a1+a2 #incident angle
        if th2 == angle1: #incoming angle is largest
            if b >= pi/2: #incident angle is large
                b2 = pi - b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th1
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
            else: #incident angle is good
                b2 = b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th1-pi
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
        else:
            if b >= pi/2: #incident angle is large
                b2 = pi - b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th2
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
            else: #incident angle is good
                b2 = b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th2-pi
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
    elif th2 > pi and th2<3*pi/2 and th1 < pi and th1 > pi/2:
        a1 = th2- pi#one angle
        a2 = pi-th1
        b = a1+a2 #incident angle
        if th2 == angle1: #incoming angle is largest
            if b >= pi/2: #incident angle is large
                b2 = pi - b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th1 + pi
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
            else: #incident angle is good
                b2 = b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th1
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
        else:
            if b >= pi/2: #incident angle is large
                b2 = pi - b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th2 - pi
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
            else: #incident angle is good
                b2 = b #redef incident angle
                if b2 <= th_crit: #zippering
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr = d/sin(b2) #dispacement from collision pt
                        new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                        col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                    new_angle = th2
                else: #catastrophe
                    # r = rnd.randint(0, 1)
                    if r== 0:
                        resolve = 'catas'
    elif th2 < pi/2 and th1 < pi/2:
        b = th2-th1 #incident angle
        if th2 == angle1: #incoming angle is largest
            if b <= th_crit: #zippering
                resolve = 'zipper+'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th1
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
        else:
            if b <= th_crit: #zippering
                resolve = 'zipper+'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th2
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
    elif th2 > pi and th2< 3*pi/2 and th1 < pi/2 and (th2-pi)>th1:
        a1 = th2-pi#one angle
        a2 = th1
        b = a1-a2#incident angle
        if th2 == angle1: #incoming angle is largest
            if b <= th_crit: #zippering
                resolve = 'zipper-'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th1 + pi
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
        else:
            if b <= th_crit: #zippering
                resolve = 'zipper-'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th2 - pi
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
    elif th2 > pi and th2< 3*pi/2 and th1 < pi/2 and (th2-pi)<th1:
        a1 = th1#one angle
        a2 = th2-pi
        b = a1-a2#incident angle
        if th2 == angle1: #incoming angle is largest
            if b <= th_crit: #zippering
                resolve = 'zipper-'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th1 + pi
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
        else:
            if b <= th_crit: #zippering
                resolve = 'zipper-'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th2 - pi
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
    elif th2 > pi and th2< 3*pi/2 and th1 > pi and th1< 3*pi/2:
        b = th2-th1#incident angle
        if th2 == angle1: #incoming angle is largest
            if b <= th_crit: #zippering
                resolve = 'zipper+'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th1
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
        else:
            if b <= th_crit: #zippering
                resolve = 'zipper+'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th2
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
    elif th2 > pi/2 and th2< pi and th1 > pi/2 and th1 < pi:
        b = th2-th1#incident angle
        if th2 == angle1: #incoming angle is largest
            if b <= th_crit: #zippering
                resolve = 'zipper+'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th1
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
        else:
            if b <= th_crit: #zippering
                resolve = 'zipper+'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th2
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
    elif th2 > 3*pi/2 and th1 > pi/2 and th1 < pi and (th1+pi)>th2:
        a1 = th1+pi#one angle
        a2 = th2
        b = a1-a2#incident angle
        if th2 == angle1: #incoming angle is largest
            if b <= th_crit: #zippering
                resolve = 'zipper-'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th1 + pi
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
        else:
            if b <= th_crit: #zippering
                resolve = 'zipper-'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th2 - pi
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
    elif th2 > 3*pi/2 and th1 > pi/2 and th1 < pi and (th1+pi)<th2:
        a1 = th2#one angle
        a2 = th1+pi
        b = a1-a2#incident angle
        if th2 == angle1: #incoming angle is largest
            if b <= th_crit: #zippering
                resolve = 'zipper-'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th1 + pi
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
        else:
            if b <= th_crit: #zippering
                resolve = 'zipper-'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th2 - pi
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
    elif th2 > 3*pi/2 and th1 > 3*pi/2:
        a1 = th2#one angle
        a2 = th1
        b = a1-a2#incident angle
        if th2 == angle1: #incoming angle is largest
            if b <= th_crit: #zippering
                resolve = 'zipper+'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th1
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
        else:
            if b <= th_crit: #zippering
                resolve = 'zipper+'
                if no_bdl_id:
                    dr = d/sin(b) #dispacement from collision pt
                    new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
                    col_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)]
                new_angle = th2
            else: #catastrophe
                # r = rnd.randint(0, 1)
                if r== 0:
                    resolve = 'catas'
    # else:
    #     print('ANGLES GONE WEIRD')
    #     print(th1/pi, th2/pi)
    # print('NEW ANGLE', new_angle/pi)
    # if dr is not None and dr<0: #check for possible error
    #     print('Zippering error', angle1,angle2)
    #     assert dr >0
    # x = new_pt[0]
    # y = new_pt[1]
    # seg_dist = dist(pt_prev,pt) #distance from vertex, don't want it to step back too much
    error = None #error message for specia cases
    # if resolve not in ['catas', 'cross']:
    #     if (y>1 or y<0 or x>1 or x<0 or dr > dr_tol or seg_dist<dr): #somewhat adhoc way of avoiding near-parallel collisions
    #         if y>1 or y<0 or x>1 or x<0:
    #             error = 'outside'
    #         elif dr > dr_tol:
    #             error = 'tol_'+str(dr)
    #         else:
    #             error = 'dist_'+str(abs(dr/seg_dist))
    #         resolve = 'special'
    #         new_pt = None #revert back to normal pt
    #         if abs(dr-seg_dist)<1e-10:
    #             print('ERROR')
    #             sys.exit()
    return(new_angle, new_pt, resolve, col_pt,error)

# def zip_cat(angle1,angle2,pt):
#     '''
#     Parameters
#     ----------
#     angle1 : angle of tip which collides
#     angle2 : angle of barrier MT
#     pt : point of intersection

#     Returns
#     -------
#     None.

#     '''
#     th2,th1 = max(angle1,angle2),min(angle1,angle2)
#     # print(angle1/pi,angle2/pi)
#     d = 0 #distance away from MT for zippering
#     th_crit = 2*pi/9 #critical angle
#     new_pt = pt
#     new_angle = angle1
#     if th2 > 3*pi/2 and th1<pi/2:
#         a1 = 2*pi-th2 #one angle
#         a2 = th1
#         b = a1+a2 #incident angle
#         if th2 == angle1: #incoming angle is largest
#             if b >= pi/2: #incident angle is large
#                 b2 = pi - b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = pi + th1
#                 # else: #catastrophe TODO
#                 #     r = rnd.randint(0, 1)
#                 #     if r== 0:
#                 #         new_angle =
#             else: #incident angle is good
#                 b2 = b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th1
#         else:
#             if b >= pi/2: #incident angle is large
#                 b2 = pi - b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th2-pi
#                 # else: #catastrophe TODO
#             else: #incident angle is good
#                 b2 = b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th2
#     elif th2 > pi/2 and th2< pi and th1<pi/2:
#         a1 = pi-th2 #one angle
#         a2 = th1
#         b = a1+a2 #incident angle
#         if th2 == angle1: #incoming angle is largest
#             if b >= pi/2: #incident angle is large
#                 b2 = pi - b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th1
#                 # else: #catastrophe TODO
#                 #     r = rnd.randint(0, 1)
#                 #     if r== 0:
#                 #         new_angle =
#             else: #incident angle is good
#                 b2 = b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th1+pi
#         else:
#             if b >= pi/2: #incident angle is large
#                 b2 = pi - b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th2
#                 # else: #catastrophe TODO
#             else: #incident angle is good
#                 b2 = b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th2+pi
#     elif th2 > 3*pi/2 and th1<3*pi/2 and th1 > pi:
#         a1 = th1- pi#one angle
#         a2 = 2*pi - th2
#         b = a1+a2 #incident angle
#         if th2 == angle1: #incoming angle is largest
#             if b >= pi/2: #incident angle is large
#                 b2 = pi - b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th1
#                 # else: #catastrophe TODO
#                 #     r = rnd.randint(0, 1)
#                 #     if r== 0:
#                 #         new_angle =
#             else: #incident angle is good
#                 b2 = b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th1-pi
#         else:
#             if b >= pi/2: #incident angle is large
#                 b2 = pi - b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th2
#                 # else: #catastrophe TODO
#             else: #incident angle is good
#                 b2 = b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th2-pi
#     elif th2 > pi and th2<3*pi/2 and th1 < pi and th1 > pi/2:
#         a1 = th2- pi#one angle
#         a2 = pi-th1
#         b = a1+a2 #incident angle
#         if th2 == angle1: #incoming angle is largest
#             if b >= pi/2: #incident angle is large
#                 b2 = pi - b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th1 + pi
#                 # else: #catastrophe TODO
#                 #     r = rnd.randint(0, 1)
#                 #     if r== 0:
#                 #         new_angle =
#             else: #incident angle is good
#                 b2 = b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th1
#         else:
#             if b >= pi/2: #incident angle is large
#                 b2 = pi - b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th2 - pi
#                 # else: #catastrophe TODO
#             else: #incident angle is good
#                 b2 = b #redef incident angle
#                 if b2 <= th_crit: #zippering
#                     new_pt = [pt[0] - dr*cos(angle1),pt[1] - dr*sin(angle1)] #new point
#                     new_angle = th2
#     # else:
#     #     print('ANGLES GONE WEIRD')
#     # print('NEW ANGLE', new_angle/pi)
#     return(new_angle, new_pt)

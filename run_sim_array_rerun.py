#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tim

Main simulation script, main_sim tries the simulation function.
To run, type into command line:
python date seed start_idx final_idx trouble_bool
See below for description for each argument.
"""
from sim_algs_fixed_region import simulate, rerun #this fn calls various others in the dependencies
import sys
# import inspect
import logging
from parameters import verbose, plot

#these will be used a arguments for the simulation function
date = str(sys.argv[1]) #unlike multprocessing script, bash inputs the date - not python
path = '../'+date+'/'
seed = int(sys.argv[2]) #bash also inputs seed
start_idx = int(sys.argv[3]) #starting index
final_idx = int(sys.argv[4])
trouble_bool = (str(sys.argv[5])=='True') #whether to save more checkpoints

def main_sim(seed, start_idx, final_idx, save_path, verbose, plot, trouble_bool): #for handling errors
    '''
    Runs the main simulation. If the starting_idx (description below) is nonzero, it will try to find the checkpoint file in the ../date/states/
    directory to load and continue from the nonzero starting time. Otherwise, it will start from scratch.
    
    You must create a ./save_path/states/ directory.
    
    Parameters
    ----------
    seed : Int
        Random number generator seed.
    start_idx : Int
        Starting hour (index starts at 0).
    final_idx : Int
        Final hour (when the simulation exits).
    save_path : String
        Root directory to save in.
    verbose : Bool
        Option to print excessive statements.
    plot : Bool
        Whether to plot the MTs at the stopping time.
    trouble_bool : Bool
        Troubleshooting mode; whether to save checkpoints more frequently.

    Returns
    -------
    Hourly checkpoint pickle files in ./save_path/states/
    
    Hourly order parameter pickle files in ./save_path/
    
    The order parameters are calculated every 10th of the hour and stored in an array i.e. each pickle file contains
    order parameter info for 10 time points between each hour.
    
    Script also save some misc. info hourly in ./save_path/
    
    If there is an error, an error .log file is created in ./save_path/
    '''
    try:
        if start_idx == -1: #TODO: when working with indices rather than hr, change to == -1
            simulate(seed, final_idx, save_path, verbose, plot, troubleshoot = trouble_bool)
        else:
            rerun(seed, start_idx, final_idx, save_path, verbose, troubleshoot = trouble_bool)
    except:
        # tau = inspect.trace()[-1][0].f_locals['tau'] #get sim time of error (hr)
        tau = sys.exc_info()[2].tb_next.tb_frame.f_locals['tau']
        logging.basicConfig(level=logging.DEBUG, filename=path+'ERROR_seed'+str(seed)+'.log')
        logging.exception("Failed at seed " + str(seed)+', time (hr) '+str(tau))

#create dir for results
print('Simulation started for ' + path,'\n')
#call simulation
main_sim(seed, start_idx, final_idx, path, verbose, plot, trouble_bool)

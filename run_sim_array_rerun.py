#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main simulation script.

There are other parameters set in the `parameters.py` file.
"""

import argparse
import logging
import sys
# import inspect

from sim_algs_fixed_region import simulate, rerun #this fn calls various others in the dependencies
from parameters import verbose, plot

__version__ = "0.0.1"


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
        Starting hour (index starts at 0).  A special value of -1 starts
        a new simulation run, rather than continuing an old one.
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


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="\n".join(__doc__.split("\n")[1:]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )

    # TODO: why is this called "date"?
    parser.add_argument("date", type=str, help="""
        Used to construct a directory to store the simulation results.
        Different simulations should a different value.
    """)
    parser.add_argument("seed", type=int, help="""
        Integer seed for random number generators.  In theory using the
        same seed should produce the same results, although this is not
        yet stable between software versions, Python versions, OS
        versions, architectures, etc.
    """)
    parser.add_argument("start_idx", type=int, help="""
        Starting hour (index starts at 0).
        A special value of -1 starts a new simulation run, rather than
        continuing an old one.
    """)
    parser.add_argument("final_idx", type=int, help="""
        Final hour (when the simulation exits).
    """)
    parser.add_argument(
        "--trouble",
        default=None,
        action="store_true",
        help="Save additional checkpoints, for example for debugging."
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    date = args.date
    path = '../'+date+'/'
    seed = args.seed
    start_idx = args.start_idx
    final_idx = args.final_idx
    trouble_bool = args.trouble

    #create dir for results
    print('Simulation started for ' + path,'\n')
    #call simulation
    main_sim(seed, start_idx, final_idx, path, verbose, plot, trouble_bool)

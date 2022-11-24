"""Main module

Contains main function that run the program

Copyright by Laura Tiessler-Sala

"""

import os
import re
import multiprocessing
import psutil
import copy
import csv
import pyKVFinder
import numpy as np
from numpy import  zeros, float, array, dot, outer, argsort, linalg, identity
import itertools
import mdtraj as md 
from sklearn.cluster import KMeans

from .additional_functions import DIST_PROBE_ALPHA, DIST_PROBE_BETA, ANGLE_PAB, DIST_PROBE_OXYGEN, ANGLE_POC, _parse_molecule, _print_pdb, print_file, _check_actual_motif, _check_possible_mutations, _calculate_center_and_radius

input = '2yoo.pdb'
pdb_filename = 'output_' + input 
pdb_file, xyz_pdb, traj_pdb, top_pdb = load_pdb(input,path_files)
dic_output_volumes = volume_pyKVFinder(pdb_file, input,path_files)
list_out_elip, list_cav = elipsoid(input, dic_output_volumes,path_files)
run_biometall(input, min_coordinators=1, min_sidechain=1,
                residues='[CYS]', motif='', grid_step=1.0,
                cluster_cutoff=0.0, pdb=False,
                propose_mutations_to='', custom_radius=None, custom_center=None,
                cores_number=None, backbone_clashes_threshold=1.0, sidechain_clashes_threshold=0.0, cmd_str="")
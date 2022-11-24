"""Main module

Contains main function that run the program

Copyright by Laura Tiessler-Sala

"""

import os
from .utils.volume_elipsoid volume_pyKVFinder, elipsoid


input = '2yoo.pdb'
pdb_filename = 'output_' + input 

#Load pdb for cavity detection
pdb_file, xyz_pdb, traj_pdb, top_pdb = load_pdb(input,path_files)

#Detect cavities and calculate volumen
dic_output_volumes = volume_pyKVFinder(pdb_file, input,path_files)

#Make ellipsoid
list_out_elip, list_cav = elipsoid(input, dic_output_volumes,path_files)

#Detect close residues

#Run biometall
run_biometall(input, min_coordinators=1, min_sidechain=1,
                residues='[CYS]', motif='', grid_step=1.0,
                cluster_cutoff=0.0, pdb=False,
                propose_mutations_to='', custom_radius=None, custom_center=None,
                cores_number=None, backbone_clashes_threshold=1.0, sidechain_clashes_threshold=0.0, cmd_str="")

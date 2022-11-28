"""Main module

Contains main function that run the program

Copyright by Laura Tiessler-Sala

"""

import os
from utils.parser import read_pdb
from utils.volume_elipsoid import volume_pyKVFinder, elipsoid
from utils.metal import run_biometall


path_files = '/HDD/3rd_year/hemefinder/hemefinder'

input = './2yoo.pdb'
pdb_id, extension = os.path.splitext(os.path.basename(input))
print(pdb_id)
#Load pdb for cavity detection
atomic = read_pdb(input)

#Detect cavities and calculate volumen
dic_output_volumes = volume_pyKVFinder(atomic, pdb_id, path_files)

#Make ellipsoid
list_out_elip, list_cav = elipsoid(pdb_id, dic_output_volumes,path_files)

#Detect close residues

#Run biometall
run_biometall(input,pdb_id, list_cav, min_coordinators=1, min_sidechain=1,
                residues='[CYS]', motif='', grid_step=1.0,
                cluster_cutoff=0.0, pdb=False,
                propose_mutations_to='', custom_radius=None, custom_center=None,
                cores_number=None, backbone_clashes_threshold=1.0, sidechain_clashes_threshold=0.0, cmd_str="")

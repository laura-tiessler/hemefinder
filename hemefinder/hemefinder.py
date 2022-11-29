"""Main module

Contains main function that run the program

Copyright by Laura Tiessler-Sala

"""
import os
import time

from .utils.metal import run_biometall
from .utils.parser import read_pdb
from .utils.volume_elipsoid import elipsoid, volume_pyKVFinder, detect_residues


def hemefinder(
    target: str,
    outputdir: str
):
    start = time.time()
    # Load pdb for cavity detection
    pdb_path = os.path.join(outputdir, target)
    atomic = read_pdb(target, outputdir)

    # Detect cavities and calculate volumen
    output_cav = volume_pyKVFinder(atomic, target, outputdir)


    # # Run biometall
    # run_biometall(
    #     pdb_path, list_cav, min_coordinators=1, min_sidechain=1,
    #     residues='[CYS]', motif='', grid_step=1.0,
    #     cluster_cutoff=0.0, pdb=False,
    #     propose_mutations_to='', custom_radius=None, custom_center=None,
    #     cores_number=None, backbone_clashes_threshold=1.0,
    #     sidechain_clashes_threshold=0.0, cmd_str=""
    # )
   
    # # Make ellipsoid
    # list_out_elip, list_cav = elipsoid(target, dic_output_volumes, outputdir)
    

    # # Detect close residues
    #residues = detect_residues(xyz, alphas, betas, res_for_column, name_for_res)

    end = time.time()
    print(f'\nComputation took {round(end - start, 2)} s')
if __name__ == '__main__':
    help(hemefinder)

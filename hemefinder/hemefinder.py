"""Main module

Contains main function that run the program

Copyright by Laura Tiessler-Sala

"""
import os
import time
from pathlib import Path

import numpy as np

from .utils.data import load_stats, load_stats_res
from .utils.parser import parse_residues, read_pdb
from .utils.print import create_PDB
from .utils.scoring import clustering, coordination_score, clustering, centroid, residue_scoring
from .utils.volume_elipsoid import detect_ellipsoid, detect_residues, elipsoid, volume_pyKVFinder


def hemefinder(
    target: str,
    outputdir: str,
    coordinators: int or list
):
    start = time.time()

    # Load stats for bimodal distributions
    stats = load_stats()
    stats_res = load_stats_res()

    # Read protein and find possible coordinating residues
    atomic = read_pdb(target, outputdir)
    alphas, betas, res_number_coordinators, all_alphas, all_betas, residues_names = parse_residues(atomic, coordinators, stats)

    # Detect cavities and analyse possible coordinations
    probes = volume_pyKVFinder(atomic, target, outputdir)
    scores = coordination_score(alphas, betas, stats, probes, res_number_coordinators)
    coord_residues = clustering(scores)
    coord_residues_centroid = centroid(coord_residues) 
    
    sorted_results = {k:v for k,v in sorted(coord_residues_centroid.items(), key=lambda x: x[1]['score'], reverse=True)}
    final_results={}

    for k, v in sorted_results.items():
        sphere = detect_ellipsoid(probes, v['centroid'])
        yes_no = elipsoid(sphere)
        if yes_no == True:
            final_results[k] = v
            final_results[k]['elipsoid'] = np.array(sphere)
            residues = detect_residues(sphere, all_alphas, all_betas, residues_names)
            score = residue_scoring(residues,stats_res)
            final_results[k]['score_res'] = score
    for k,v in final_results.items():
        print(k, v['elipsoid'])
    basename = Path(target).stem
    outputfile = os.path.join(outputdir, basename)
    print(outputfile)
    create_PDB(final_results, outputfile, 'centroid')
    create_PDB(final_results, outputfile, 'elipsoid')
    create_PDB(final_results, outputfile, 'probes')


        
    end = time.time()
    print(f'\nComputation took {round(end - start, 2)} s')


if __name__ == '__main__':
    help(hemefinder)

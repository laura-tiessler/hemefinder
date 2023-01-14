"""Main module

Contains main function that run the program

Copyright by Laura Tiessler-Sala

"""
import os
import time
from pathlib import Path
from unittest import result

import numpy as np

from .utils.data import load_stats, load_stats_res
from .utils.parser import parse_residues, read_pdb
from .utils.print import create_PDB
from .utils.scoring import clustering, coordination_score, clustering, centroid, residue_scoring, new_probes
from .utils.volume_elipsoid import detect_ellipsoid, detect_residues, elipsoid, volume_pyKVFinder, detect_hole


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
    results_by_cluster = {}

    for i, probe in enumerate(probes):
        final_results={}
        dic_coordinating = {}
        scores = coordination_score(alphas, betas, stats, probe, res_number_coordinators)
        coord_residues = clustering(scores, dic_coordinating)
        if len(coord_residues) == 0:
            continue
        coord_residues_centroid = centroid(coord_residues)
        
        for k, v in coord_residues_centroid.items():
            sphere = detect_ellipsoid(probe, v['centroid'])
            yes_no = elipsoid(sphere)
            if yes_no == True:
                final_results[k] = v
                final_results[k]['elipsoid'] = np.array(sphere)
                residues = detect_residues(sphere, all_alphas, all_betas, residues_names)
                score = residue_scoring(residues,stats_res)
                final_results[k]['score_res'] = score
        results_by_cluster[i] = final_results

    basename = Path(target).stem + '_inicial'
    outputfile = os.path.join(outputdir, basename)
    
    final_dic = {}
    results = results_by_cluster.values()
    for res in results_by_cluster.values():
        final_dic.update(res)

    sorted_results =  {k:v for k,v in sorted(final_dic.items(), key=lambda x: x[1]['score'], reverse=True)}
    num = 1
    for k,v in sorted_results.items():
        print(num, k, v['score'])
        num += 1
    
    create_PDB(sorted_results, outputfile)

        
    end = time.time()
    print(f'\nComputation took {round(end - start, 2)} s')


if __name__ == '__main__':
    help(hemefinder)

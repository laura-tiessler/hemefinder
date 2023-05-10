"""Main module

Contains main function that run the program

Copyright by Laura Tiessler-Sala

"""
import os
import time
from pathlib import Path
from unittest import result

import numpy as np
import json
from json import dumps

from .utils.data import load_stats, load_stats_res
from .utils.parser import parse_residues, read_pdb
from .utils.print import create_PDB
from .utils.scoring import centroid_elipsoid, clustering, coordination_score, clustering, centroid, residue_scoring, new_probes, coordination_score_mutation, clustering_mutation
from .utils.volume_elipsoid import detect_ellipsoid, detect_residues, elipsoid, volume_pyKVFinder, detect_hole


def hemefinder(
    target: str,
    outputdir: str,
    coordinators: int or list,
    mutations: int or list
):
    start = time.time()
    # Load stats for bimodal distributions
    stats = load_stats()
    stats_res = load_stats_res()

    # Read protein and find possible coordinating residues
    atomic = read_pdb(target, outputdir)
    alphas, betas, res_name_number_coord, all_alphas, all_betas, residues_names, residues_ids = parse_residues(target, atomic, coordinators, stats)

    # Detect cavities and analyse possible coordinations

    probes = volume_pyKVFinder(atomic, target, outputdir)
    results_by_cluster = {}

    #Not mutations
    if len(mutations) == 0:
        for i, probe in enumerate(probes):
            final_results={}
            dic_coordinating = {}
            scores = coordination_score(alphas, betas, stats, probe, res_name_number_coord)
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
                    score_res = residue_scoring(residues,stats_res)
                    final_results[k]['score_res'] = score_res
                    score_eli = centroid_elipsoid(v['centroid'],v['elipsoid'])
                    final_results[k]['score_elipsoid'] = score_eli
                    final_results[k]['total_score'] = v['score']+(score_res*score_eli)
            results_by_cluster[i] = final_results
    
    #For mutations
    else:
        for i, probe in enumerate(probes):
            final_results={}
            dic_coordinating = {}
            scores = coordination_score_mutation(all_alphas, all_betas, stats, probe, res_name_number_coord, mutations, residues_ids)
            coord_residues = clustering_mutation(scores, dic_coordinating, mutations)
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


    basename = Path(target).stem 
    outputfile = os.path.join(outputdir, basename)
    
    final_dic = {}
    for res in results_by_cluster.values():
        final_dic.update(res)

    if len(mutations) == 0:
        sorted_results =  {k:v for k,v in sorted(final_dic.items(), key=lambda x: x[1]['total_score'], reverse=True)}
        num = 1
        for k,v in sorted_results.items():
            print(num, k, v['total_score'],v['score'])
            num += 1
        create_PDB(sorted_results, outputfile)
        sorted_results_str = {str(k): v['total_score'] for k, v in sorted_results.items()}
        json_object = json.dumps(sorted_results_str, indent=4)
        out_json =  outputfile + '.json'
        with open(out_json, "w") as outfile_json:
            outfile_json.write(json_object)

    # else:
    #     for k, v in final_dic.items():
    #         if v['score']['HIS'] > 10:
    #             print(k, v['score'])

        
    end = time.time()
    print(f'\nComputation took {round(end - start, 2)} s')


if __name__ == '__main__':
    help(hemefinder)

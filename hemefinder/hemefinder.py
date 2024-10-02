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

from .utils.data import load_stats, load_stats_res, load_stats_two_coord
from .utils.parser import parse_residues, read_pdb
from .utils.print import create_PDB
from .utils.scoring import (
    centroid_elipsoid,
    clustering,
    coordination_score,
    clustering,
    centroid,
    residue_scoring,
    new_probes,
    clustering_mutation,
    two_coordinants,
)
from .utils.volume_elipsoid import (
    detect_ellipsoid,
    detect_residues,
    elipsoid,
    volume_pyKVFinder,
    detect_hole,
)


def normalize(value, maxim, minim, diff):
    if diff == 0:
        return 1
    normal = float((value - minim) / diff)
    return normal


def hemefinder(
        target: str, outputdir: str, coordinators: str, min_num_coordinators: int, mutations: list, probe_in: float, 
    probe_out: float, removal_distance = float, volume_cutoff = float, surface = str
):
    start = time.time()
    # Load stats for bimodal distributions
    stats = load_stats()
    stats_res = load_stats_res()
    stats_two_coord = load_stats_two_coord()

    #Transform input to list for residues and mutation
    coordinators = list(map(str, coordinators.strip('[]').split(',')))

    # Read protein and find possible coordinating residues
    atomic = read_pdb(target, outputdir)
    (
        alphas,
        betas,
        res_name_number_coord,
        all_alphas,
        all_betas,
        residues_names,
        residues_ids,
    ) = parse_residues(target, atomic, coordinators, stats)
    # Detect cavities and analyse possible coordinations
    probes = volume_pyKVFinder(atomic, target, outputdir, probe_in, probe_out, removal_distance, volume_cutoff, surface)
    results_by_cluster = {}

    # all_probes = np.concatenate(probes)

    if len(mutations) != 0:
        alphas = all_alphas
        betas = all_betas

    for i, probe in enumerate(probes):  # Loop throug different cluster and its proves
        final_results = {}
        dic_coordinating = {}
        scores = coordination_score(
            alphas,
            betas,
            stats,
            probe,
            res_name_number_coord,
            mutations,
            residues_ids,
        )
        coord_residues = clustering(scores, dic_coordinating, mutations)
        if len(coord_residues) == 0:
            continue
        coord_residues_centroid = centroid(coord_residues)

        for k, v in coord_residues_centroid.items():
            sphere = detect_ellipsoid(probe, v["centroid"])
            axes, d2, center, elen = elipsoid(sphere)
            if elen != 0:
                final_results[k] = v
                final_results[k]["elipsoid"] = np.array(sphere)
                residues = detect_residues(
                    sphere, all_alphas, all_betas, residues_names
                )
                score_res = residue_scoring(residues, stats_res)
                final_results[k]["score_res"] = score_res
                score_eli = centroid_elipsoid(
                    v["centroid"], v["elipsoid"], elen, axes, center, d2
                )
                final_results[k]["score_elipsoid"] = score_eli
        results_by_cluster[i] = final_results

    basename = Path(target).stem
    outputfile = os.path.join(outputdir, basename)

    final_dic = {}
    # Loop through all the results of different clusters and selects the one that has the highest score
    for res in results_by_cluster.values():
        for res_ind, res_ind_values in res.items():
            if res_ind not in final_dic:
                final_dic.update({res_ind: res_ind_values})
            else:
                if res_ind_values["score"] > final_dic[res_ind]["score"]:
                    final_dic.update({res_ind: res_ind_values})

    # Check cases two coordinants or minimum cordinants is correct:
    to_remove = []
    for residues_coord, residues_coord_values in final_dic.items():
        if len(residues_coord) > 1:
            two_coord_veredict = two_coordinants(
                all_alphas,
                all_betas,
                residues_coord,
                residues_coord_values["centroid"],
                residues_ids,
                residues_names,
                stats_two_coord,
            )
            if two_coord_veredict == "no":
                to_remove.append(residues_coord)
        else:
            if min_num_coordinators == 2:
                to_remove.append(residues_coord)

    final_dic = {key: final_dic[key] for key in final_dic if key not in to_remove}

    scores = [val["score"] for key, val in final_dic.items() if "score" in val]
    max_scores = max(scores)
    min_score = min(scores)
    diff_score = max_scores - min_score
    scores_eli = [
        val["score_elipsoid"]
        for key, val in final_dic.items()
        if "score_elipsoid" in val
    ]
    max_scores_eli = max(scores_eli)  # inversed
    min_score_eli = min(scores_eli)  # inversed
    diff_score_eli = max_scores_eli - min_score_eli
    scores_res = [
        val["score_res"] for key, val in final_dic.items() if "score_res" in val
    ]
    max_scores_res = max(scores_res)
    min_score_res = min(scores_res)
    diff_score_res = max_scores_res - min_score_res

    for k, v in final_dic.items():
        score_norm = normalize(final_dic[k]["score"], max_scores, min_score, diff_score)
        score_norm_res = normalize(
            final_dic[k]["score_res"], max_scores_res, min_score_res, diff_score_res
        )
        score_norm_eli = normalize(
            final_dic[k]["score_elipsoid"],
            max_scores_eli,
            min_score_eli,
            diff_score_eli,
        )
        final_dic[k]["total_score_norm"] = score_norm + score_norm_eli + score_norm_res

    sorted_results = {
        k: v
        for k, v in sorted(
            final_dic.items(), key=lambda x: x[1]["total_score_norm"], reverse=True
        )
    }
    num = 1
    for k, v in sorted_results.items():
        print(
            num,
            k,
            v["total_score_norm"],
        )
        num += 1
    create_PDB(sorted_results, outputfile)
    sorted_results_str = {
        str(k): v["total_score_norm"] for k, v in sorted_results.items()
    }
    json_object = json.dumps(sorted_results_str, indent=4)
    out_json = outputfile + ".json"
    with open(out_json, "w") as outfile_json:
        outfile_json.write(json_object)
    
    for fname in os.listdir(outputdir):
        if fname.startswith("cavity") or fname.startswith("cluster"):
            os.remove(os.path.join(outputdir, fname))

    end = time.time()
    f = open("./time.txt", "a")
    final_time = round(end - start, 2)
    f.write(str(target[:-4]))
    f.write(str(":"))
    f.write(str(final_time))
    f.write(str("\n"))
    f.close()
    print(f"\nComputation took {round(end - start, 2)} s")


if __name__ == "__main__":
    help(hemefinder)

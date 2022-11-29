"""Main module

Contains main function that run the program

Copyright by Laura Tiessler-Sala

"""
import os
import time
from pathlib import Path

import numpy as np

from .utils.data import load_stats
from .utils.parser import parse_residues, read_pdb
from .utils.print import create_PDB
from .utils.scoring import coordination_analysis
from .utils.volume_elipsoid import detect_residues, elipsoid, volume_pyKVFinder


def hemefinder(
    target: str,
    outputdir: str,
    coordinators: int or list
):
    start = time.time()

    # Load stats for bimodal distributions
    stats = load_stats()

    # Read protein and find possible coordinating residues
    atomic = read_pdb(target, outputdir)
    alphas, betas = parse_residues(atomic, coordinators, stats)

    # Detect cavities and analyse possible coordinations
    probes = volume_pyKVFinder(atomic, target, outputdir)
    scores = coordination_analysis(alphas, betas, stats, probes)

    best_scores = np.argwhere(scores[:, 3] > 0.5)
    new_scores = np.zeros((len(best_scores), 4))
    for idx, idx_score in enumerate(best_scores):
        new_scores[idx, :] = scores[idx_score, :]

    basename = Path(target).stem
    outputfile = os.path.join(outputdir, basename)
    create_PDB(new_scores, outputfile)

    # # Make ellipsoid
    # list_out_elip, list_cav = elipsoid(target, dic_output_volumes, outputdir)

    # # Detect close residues
    # residues = detect_residues(xyz, alphas, betas, res_for_column, name_for_res)

    end = time.time()
    print(f'\nComputation took {round(end - start, 2)} s')


if __name__ == '__main__':
    help(hemefinder)

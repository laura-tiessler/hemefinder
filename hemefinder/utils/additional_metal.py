import itertools
from collections import Counter
from typing import List, Tuple

import numpy as np


def _calculate_center_and_radius(probes) -> Tuple[np.ndarray, float]:
    """
    Calculates the most central point of a list and its distance to the furthest
    point.
    Euclidean distances from all the points to all the points of the list (i.e.
    `probes`) are calculated (i.e. distance_matrix). Then, the point with the
    smallest sum of distances is considered the most central. The distance from
    this central point to its farthest would be the radius which is necessary to
    embed all the points in a sphere.
    Parameters
    ----------
    probes : list of array_like
        List of 3-float arrays containing coordinates
    Returns
    -------
    array_like
        3-float array containing the coordinates of the most central point
    float
        radius of the sphere which is necessary to embed all points
    """
    probes = np.array(probes)

    # 1. Euclidean distance from all to all probes
    distance_matrix = np.linalg.norm(probes - probes[:,None], axis=-1)

    # 2. Search for the most central point of the list
    for i, probe in enumerate(distance_matrix):
        sum_dist_probe = sum(probe)  # Sum of distances to all the other probes
        if i == 0:
            min_sum_dist = sum_dist_probe
            best_probe = i
            highest_dist = max(probe)
        else:
            if sum_dist_probe < min_sum_dist:
                min_sum_dist = sum_dist_probe
                best_probe = i
                highest_dist = max(probe)
    return probes[best_probe], highest_dist


def _counterSubset(list1, list2) -> bool:
    """
    Check if all the elements of list1 are contained in list2.

    It counts the quantity of each element (i.e. 3-letter amino acid code) 
    in the list1 (e.g. two 'HIS' and one 'ASP') and checks if the list2 contains 
    at least that quantity of every element.
    Parameters
    ----------
    list1 : list
        List of amino acids in 3-letter code (can be repetitions)
    list2 : list
        List of amino acids in 3 letter code (can be repetitions)
    Returns
    -------
    bool
        True if list2 contains all the elements of list1. False otherwise
    """
    # 1. Count how many amino acids of each kind
    c1, c2 = Counter(list1), Counter(list2)

    # 2. Check that list2 contains at least the same number of every amino acid
    for k, n in c1.items():
        if n > c2[k]:
            return False
    return True


def _check_actual_motif(motif_possibilities, probe_coordinators, name_for_res, 
                        res_for_column) -> List[str]:
    """
    Checks if the coordinators of a probe match the motif requested by the user.
    A motif is matched when the possible amino acids that coordinate a probe
    include all the contained in the motif. Only the actual amino acids present 
    in the structure provided by the user are considered. For example, a motif
    [HIS,HIS,ASP/GLU] would be accomplished if the coordinators of the probe 
    contains, at least, either (HIS,HIS,ASP) or (HIS,HIS,GLU).
    The function returns all the possible combinations of amino acids that match
    the motif for the given probe.
    Parameters
    ----------
    motif_possibilities : list of sequences
        All the combinations of amino acids that accomplish the motif
    probe_coordinators : dict
        Contains the amino acids that coordinate the probe. Indexed by number of
        column, the values are the names of the coordinating amino acids.
    name_for_res : dict
        Correspondence between number:chain of residue and residue name
    res_for_column: dict
        Correspondence between column number and number:chain of residue
    Returns
    -------
    list of sequences
        sequences of the amino acids that match the motif for the queried probe
    """
    c = []
    for i in list(probe_coordinators):
        if name_for_res[res_for_column[i]] in probe_coordinators[i]:
            c.append(tuple([name_for_res[res_for_column[i]], i]))
    solutions = set()
    for m in list(itertools.combinations(c, len(motif_possibilities[0]))):
        for possible_motif in motif_possibilities:
            if (_counterSubset(possible_motif, [m_tuple[0] for m_tuple in m])):
                solutions.add(tuple(m_tuple[1] for m_tuple in m))
    return solutions


def _check_possible_mutations(
    mutated_motif_possibilities,
    residues_of_mutated_motif,
    probe_coordinators
) -> list:
    """
    Calculates the possible mutations to achieve a motif requested by the user.
    A motif is matched when the possible amino acids that coordinate a probe
    include all the contained in the motif. Both the real and possible mutations
    of the amino acids are considered. For example, a motif [HIS,ASP/GLU] would 
    be accomplished if the coordinators of the probe contain, either in the 
    actual structure or by making a mutation, (HIS,ASP) or (HIS,GLU). 
    The function returns all the mutations that could be made in the residues
    to achieve the coordinating motif for the given probe.
    Parameters
    ----------
    mutated_motif_possibilities : list of sequences
        All the combinations of amino acids that accomplish the motif
    residues_of_mutated_motif : sequence of str
        Sequence of the names of the amino acids involved in the motif
    probe_coordinators : dict
        Contains the amino acids that coordinate the probe. Indexed by number of
        column, the values are the names of the coordinating amino acids and 
        their possible mutations.
    name_for_res : dict
        Correspondence between number:chain of residue and residue name
    res_for_column: dict
        Correspondence between column number and number:chain of residue
    Returns
    -------
    dict
        Possible mutations for a given residue (indexed by column number)
    """
    c = []
    for i in list(probe_coordinators):
        c.append([])
        for j in probe_coordinators[i]:
            c[-1].append(tuple([j, i]))

    c_possibilities = list(itertools.product(*c))

    mutations = {}
    for m in c_possibilities:
        for possible_motif in mutated_motif_possibilities:
            if (_counterSubset(possible_motif, [m_tuple[0] for m_tuple in m])):
                for p in probe_coordinators:
                    for r_name in probe_coordinators[p]:
                        # Exists possibility of mutation
                        if residues_of_mutated_motif.intersection(set([r_name])): 
                            if p not in list(mutations):
                                mutations[p] = set([r_name])
                            else:
                                mutations[p].add(r_name)
                break
        if mutations:
            break
    return mutations

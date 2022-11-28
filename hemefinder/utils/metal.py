import copy
import itertools
import multiprocessing
from functools import partial

import numpy as np
import psutil

from .additional_metal import (_calculate_center_and_radius,
                               _check_actual_motif, _check_possible_mutations)
from .data import ANGLE_PAB, DIST_PROBE_ALPHA, DIST_PROBE_BETA
from .parser import _parse_molecule
from .print import _print_pdb, print_file


def run_biometall(pdb_id, list_cav, min_coordinators, min_sidechain, residues, motif, grid_step,  cluster_cutoff, pdb, propose_mutations_to, custom_radius, custom_center, cores_number, backbone_clashes_threshold, sidechain_clashes_threshold, cmd_str):
    input = pdb_id + '.pdb'
    with open(input, "r") as f:
        lines = f.read().splitlines()

    #Load all the data for the pdb
    alphas, betas, carbons, nitrogens, oxygens, column_for_res, res_for_column, name_for_res, atoms_in_res, side_chains, coords_array = _parse_molecule(lines, '.pdb')
    alpha_beta_distances = np.sqrt((np.square(betas-alphas).sum(axis=1)))

    #Check that motifs are okay
    motifs, mutated_motif, residues, min_coordinators = checking_input(motif, propose_mutations_to,residues,min_coordinators)

    if not cores_number:
        cores_number = psutil.cpu_count(logical=False)
    #cores_number = 1
    pool = multiprocessing.Pool(cores_number)
    #chunks = distribute(list_cav, cores_number)
    coordination_results = pool.map(partial(_coordination, alphas=alphas,betas=betas, carbons=carbons, nitrogens=nitrogens,
        oxygens=oxygens, side_chains=side_chains, alpha_beta_distances=alpha_beta_distances, name_for_res=name_for_res, column_for_res=column_for_res,
        res_for_column=res_for_column, atoms_in_res=atoms_in_res, DIST_PROBE_ALPHA=DIST_PROBE_ALPHA, DIST_PROBE_BETA=DIST_PROBE_BETA,
        ANGLE_PAB=ANGLE_PAB,bck_clashes=1.0,sc_clashes=2.0), list_cav)
    

    centers, mutations = clustering(coordination_results, residues, motifs, min_coordinators, min_sidechain,
                       mutated_motif, cluster_cutoff, input, name_for_res, res_for_column, atoms_in_res)


    sorted_data = sorted(centers, key=lambda x: x[2], reverse=True)
    pdb_filename = f'{pdb_id}_hemefinder.pdb' 
    _print_pdb(sorted_data, pdb_filename)
    print_file(centers, motif, propose_mutations_to, mutations, name_for_res, pdb, input, cmd_str, input[:-4])


def checking_input(motif, propose_mutations_to,residues, min_coordinators):
    
    #read motif
    
    if motif:
        motifs = list(map(str, motif.strip('[]').split(',')))
        motifs_list = []
        for mot in motifs:
            motifs_list.append(list(map(str, mot.split('/'))))
        motifs = motifs_list
        motifs.sort(key=len)
    else:
        motifs = None

    if propose_mutations_to:
        mutated_motif = list(map(str, propose_mutations_to.strip('[]').split(',')))
        mutated_motif_list = []
        for mot in mutated_motif:
            mutated_motif_list.append(list(map(str, mot.split('/'))))
        mutated_motif = mutated_motif_list
        mutated_motif.sort(key=len)
    else:
        mutated_motif = None

    #Set residues to consider as coordinating
    if propose_mutations_to:
        tot = mutated_motif_list + motifs_list
        residues = list(set(x for l in tot for x in l))
    elif motif:
        tot = motifs_list
        residues = list(set(x for l in tot for x in l))
    else:
        residues = list(map(str, residues.strip('[]').split(',')))

    # Control of a correct usage of parameters
    if propose_mutations_to and motif:
        #min_coordinators should be at least len(motif) + len(mutated_motif)
        if min_coordinators < (len(motifs) + len(mutated_motif)):
            min_coordinators = len(motifs) + len(mutated_motif)
    elif propose_mutations_to and not motif:
        print("To propose mutations is necessary to set a base motif with --motif parameter.")
    elif motif:
        #min_coordinators should be at least len(motif)
        if min_coordinators < len(motifs):
            min_coordinators = len(motifs)
            print("min_coordinators has been set to {} due to the motif length".format(len(motifs)))
    return motifs, mutated_motif, residues, min_coordinators

  
  
def _coordination(grid, alphas, betas, carbons, nitrogens, oxygens, side_chains, alpha_beta_distances,name_for_res, column_for_res, 
                res_for_column, atoms_in_res, DIST_PROBE_ALPHA, DIST_PROBE_BETA, ANGLE_PAB, bck_clashes, sc_clashes):

    """
    This function calculates all the distances between atoms alpha, betas and all 
    """
    alpha_distances = np.sqrt((np.square(grid[:,np.newaxis]-alphas).sum(axis=2)))
    beta_distances = np.sqrt((np.square(grid[:,np.newaxis]-betas).sum(axis=2)))
    carbon_distances = np.sqrt((np.square(grid[:,np.newaxis]-carbons).sum(axis=2)))
    nitrogen_distances = np.sqrt((np.square(grid[:,np.newaxis]-nitrogens).sum(axis=2)))
    oxygen_distances = np.sqrt((np.square(grid[:,np.newaxis]-oxygens).sum(axis=2)))
    
    PAB_angles = np.arccos((np.square(alpha_distances) + np.square(alpha_beta_distances) - np.square(beta_distances)) / (2*alpha_distances*alpha_beta_distances))

    coords = {}
    
    # include sidechain coordinations (Alpha+Beta)
    for res_name in list(DIST_PROBE_ALPHA):
        coords[res_name] = np.dstack(np.where((DIST_PROBE_ALPHA[res_name][0]<=alpha_distances) & (alpha_distances<=DIST_PROBE_ALPHA[res_name][1]) &
                                                (DIST_PROBE_BETA[res_name][0]<=beta_distances) & (beta_distances<=DIST_PROBE_BETA[res_name][1]) &
                                                (ANGLE_PAB[res_name][0]<=PAB_angles) & (PAB_angles<=ANGLE_PAB[res_name][1])))
    # If there is a clash (distance < bck_clashes) with a backbone atom,
    # no coordination is possible for that probe
    discarded = set(np.where((oxygen_distances<bck_clashes) |
                            (carbon_distances<bck_clashes) |
                            (nitrogen_distances<bck_clashes) |
                            (alpha_distances<bck_clashes))[0])

    if sc_clashes > 0:
        for sc_atom in side_chains:
            sidechain_distances = np.sqrt((np.square(grid[:,np.newaxis]-sc_atom).sum(axis=2)))
            discarded = discarded.union(set(np.where(sidechain_distances<sc_clashes)[0]))
    
    return [grid, coords, discarded]



def clustering(coordination_results, residues, motifs, min_coordinators, min_sidechain, mutated_motif, 
                    cluster_cutoff, filename,name_for_res, res_for_column, atoms_in_res):

    dict_cluster = {}
    dict_mutations = {}
    centers = []


    #Creates all combinations of motif and mutated motifs
    if motifs:
        motif_possibilities = list(itertools.product(*motifs))
    if mutated_motif:
        mutated_motif_possibilities = list(itertools.product(*mutated_motif))
        residues_of_mutated_motif = set(x for l in mutated_motif for x in l)

    
    #Coordinations chunks is [probes from grid, coordination points {id prove: residue id}, discarded probes]
    for probes, coordinations, discarded in coordination_results:
        coordinators = {}
        for possible_coord_name in residues: #Sidechain coordinations
            for probe_idx, res_idx in coordinations[possible_coord_name][0,:,:]:
                if not mutated_motif and (possible_coord_name != name_for_res[res_for_column[res_idx]]): #discarded due to different residue name
                    continue
                if probe_idx in discarded: #residue is discarded for that probe
                    continue
                elif ('CA' not in atoms_in_res[res_for_column[res_idx]]) or ('CB' not in atoms_in_res[res_for_column[res_idx]]):
                    continue
                else: #coordination is valid for that probe and residue
                    if probe_idx not in list(coordinators):
                        coordinators[probe_idx] = {res_idx: [possible_coord_name]}
                    else:
                        if res_idx not in list(coordinators[probe_idx]):
                            coordinators[probe_idx][res_idx] = [possible_coord_name]
                        else:
                            coordinators[probe_idx][res_idx].append(possible_coord_name)
        
        for probe_idx in list(coordinators):
            if motifs:
                #A minimum motif should be accomplished without mutations (contained in motifs/motif_possibilities)
                actual_motif_solutions = _check_actual_motif(motif_possibilities, coordinators[probe_idx], name_for_res, res_for_column)
            if mutated_motif:
                #The resting coordinators of each solution should be mutable to complete the propose_mutations_to motif
                mutation_motif_solutions = {}
                for actual_motif_solution in actual_motif_solutions:
                    resting_coordinators = copy.deepcopy(coordinators[probe_idx])
                    for r in actual_motif_solution:
                        del resting_coordinators[r]
                    mutation_motif_solutions = _check_possible_mutations(mutated_motif_possibilities, residues_of_mutated_motif, resting_coordinators)

            if motifs and mutated_motif and actual_motif_solutions and mutation_motif_solutions:
                #Searching for already present motifs plus proposing mutations
                #already present motifs
                for actual_motif_solution in actual_motif_solutions:
                    item = [probes[probe_idx], tuple(sorted([res_for_column[c] for c in actual_motif_solution]))]
                    if item[1] not in list(dict_cluster):
                        dict_cluster[item[1]] = [item[0]]
                    else:
                        dict_cluster[item[1]] += [item[0]]
                    #Mutation information
                    for m in list(mutation_motif_solutions):
                        if item[1] not in dict_mutations:
                            dict_mutations[item[1]] = {res_for_column[m]: [[el, 1] for el in mutation_motif_solutions[m]]}
                        else:
                            if res_for_column[m] not in dict_mutations[item[1]]:
                                dict_mutations[item[1]][res_for_column[m]] = [[el, 1] for el in mutation_motif_solutions[m]]
                            else:
                                for el1 in mutation_motif_solutions[m]:
                                    if el1 not in [el2[0] for el2 in dict_mutations[item[1]][res_for_column[m]]]:
                                        dict_mutations[item[1]][res_for_column[m]].append([el1, 1])
                                    else:
                                        for el2 in dict_mutations[item[1]][res_for_column[m]]:
                                            if el1 == el2[0]:
                                                el2[1] += 1
                                                break
            elif motifs and not mutated_motif:
                #Searching for already present motifs
                for actual_motif_solution in actual_motif_solutions:
                    item = [probes[probe_idx], tuple(sorted([res_for_column[c] for c in actual_motif_solution]))]
                    if item[1] not in list(dict_cluster):
                        dict_cluster[item[1]] = [item[0]]
                    else:
                        dict_cluster[item[1]] += [item[0]]
            elif not motifs and not mutated_motif:
                #Searching number of coordinators
                sidechain_coord = []
                bck_coord = []
                for item in list(coordinators[probe_idx]):
                    actual_residue_name = name_for_res[res_for_column[item]]
                    if actual_residue_name in coordinators[probe_idx][item]:
                        sidechain_coord.append(res_for_column[item])
                    if actual_residue_name + "_BCK" in coordinators[probe_idx][item]:
                        bck_coord.append(str(res_for_column[item]) + "_BCK")
                if ((len(sidechain_coord) + len(bck_coord)) >= min_coordinators) and (len(sidechain_coord) >= min_sidechain):
                    c = tuple(sorted(sidechain_coord + bck_coord))
                    for coord_environment in list(dict_cluster):
                        if set(c).issuperset(coord_environment):
                            dict_cluster[coord_environment] += [probes[probe_idx]]
                    coordination_possibilities = []
                    for i in range(min_coordinators, len(c)+1):
                        coordination_possibilities += list(itertools.combinations(c,i))
                    coordination_possibilities = [tuple(sorted(el)) for el in coordination_possibilities]
                    for el in coordination_possibilities:
                        sc_num = sum("BCK" not in L for L in el)
                        if (el not in dict_cluster) and sc_num >= min_sidechain:
                            dict_cluster[el] = [probes[probe_idx]]
    
    #Order Mutation information
    for coord_environment in list(dict_mutations):
        for residue in list(dict_mutations[coord_environment]):
            dict_mutations[coord_environment][residue].sort(key = lambda x: x[1], reverse=True)   #order every residue by number of probes
    for coord_environment in list(dict_mutations):
        dict_mutations[coord_environment] = sorted(dict_mutations[coord_environment].items(), key=lambda item: item[1][0][1], reverse=True)

    try:
        max_probes = max([len(v) for v in dict_cluster.values()])
    except:
        print("None possible coordinating sites have been found. Try again with other parameters or check/change the input file.")

    for coord_residues,probes in dict_cluster.items():
        if len(probes) >= max_probes*cluster_cutoff:
            center, radius_search = _calculate_center_and_radius(probes)
            centers.append((coord_residues, center, len(probes), radius_search, probes))
    
    return centers, dict_mutations

from ossaudiodev import SOUND_MIXER_SYNTH
import numpy as np
from .additional import grid

chemical_nature = {
    'aromatic': {'PHE', 'TYR', 'TRP'},
    'polar': {'TYR', 'THR', 'SER', 'CYS', 'MET', 'ASN', 'GLN', 'HIS'},
    'positive': {'HIS', 'LYS', 'ARG'},
    'negative': {'ASP', 'GLU'},
    'hidrophobic': {'PHE', 'TRP', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO'},
    'large_chain': {'PHE', 'TYR', 'TRP', 'MET', 'GLN', 'GLU', 'ARG', 'LYS', 'HIS', 'LEU', 'ILE'}
}

def coordination_score(
    alphas: dict,
    betas: dict,
    stats: dict,
    probes: np.array,
    res_name_num: np.array):
    results_score = []
    for x, probe in enumerate(probes):
        alpha_dists = alphas - probe
        beta_dists =  betas - probe
        d1, d2, angle = geometry(alpha_dists, beta_dists)
        possible_coordinators, scores = gaussian_scoring(d1, d2, angle, stats, res_name_num)
        results_score.append([probe, possible_coordinators, scores])
    return results_score


def gaussian_scoring(
    d1: np.array,
    d2: np.array,
    angle: np.array,
    stats: dict,
    res_name_num: dict
) -> float:

    alpha_scores = []
    beta_scores = []
    angle_scores = [] 
    
    for i, res in enumerate(res_name_num[:,0]):
        alpha_scores.append(bimodal(d1[i], *stats[res]['dist_alpha']))
        beta_scores.append(bimodal(d2[i], *stats[res]['dist_beta']))
        angle_scores.append(bimodal(angle[i], *stats[res]['PAB_angle']))

    alpha_scores = np.array(alpha_scores)
    beta_scores = np.array(beta_scores)
    angle_scores = np.array(angle_scores)

    possible_coordinators = []
    ind_coordinators = np.where((alpha_scores>0.001)&(beta_scores>0.001)&(angle_scores>0.01))[0]
    if len(ind_coordinators)==0:
        fitness = 0
    else:
        fitness = 0
        for ind in ind_coordinators:
            score_ca = alpha_scores[ind]
            score_cb = beta_scores[ind]
            score_angle = angle_scores[ind]
            score_total = (score_ca + score_cb + score_angle) / 3
            possible_coordinators.append(res_name_num[ind][1])
            fitness += score_total*stats[res_name_num[ind][0]]['fitness']
    return possible_coordinators, fitness


def clustering(scores, dic_coordinating):
    for coord, residues, score in scores:
        if score ==0 or len(residues) == 0:
            continue
        else:
            residues_t = tuple(residues)
            if residues_t not in dic_coordinating:
                dic_coordinating[residues_t] = {'probes': [coord], 'score': score}
            else:
                dic_coordinating[residues_t]['probes'].append(coord)
                dic_coordinating[residues_t]['score'] += score
    return dic_coordinating


def coordination_score_mutation(
    alphas: dict,
    betas: dict,
    stats: dict,
    probes: np.array,
    res_name_num: dict,
    mutations: str,
    residues_ids:np.array):
    results_score = []
    for x, probe in enumerate(probes):
        alpha_dists = alphas - probe
        beta_dists =  betas - probe
        d1, d2, angle = geometry(alpha_dists, beta_dists)
        score_mut = gaussian_scoring_mutation(d1, d2, angle, stats, res_name_num, mutations, residues_ids)
        results_score.append([probe, score_mut])
    return results_score

def gaussian_scoring_mutation_bad(
    d1: np.array,
    d2: np.array,
    angle: np.array,
    stats: dict,
    res_name_num: dict,
    mutations: dict
) -> float:
    dic_mutations_scores = dict.fromkeys(mutations)
    
    print(dic_mutations_scores)

    for res in mutations:
        alpha_scores = bimodal(d1, *stats[res]['dist_alpha'])
        beta_scores= bimodal(d2, *stats[res]['dist_beta'])
        angle_scores = bimodal(angle, *stats[res]['PAB_angle'])

        ind_coordinators = np.where((alpha_scores>0.01)&(beta_scores>0.01)&(angle_scores>0.01))[0]
        possible_coordinators = []

        fitness = 0
        for ind in ind_coordinators:
            score_ca = alpha_scores[ind]
            score_cb = beta_scores[ind]
            score_angle = angle_scores[ind]
            score_total = (score_ca + score_cb + score_angle) / 3
            possible_coordinators.append(res_name_num[ind][1])
            fitness += score_total*stats[res]['fitness']
        dic_mutations_scores[res] = [possible_coordinators, fitness]

    print(dic_mutations_scores)
    return dic_mutations_scores

def gaussian_scoring_mutation(
    d1: np.array,
    d2: np.array,
    angle: np.array,
    stats: dict,
    res_name_num: dict,
    mutations: list,
    residues_ids: np.array
) -> float:

    dic_mutations_scores = dict.fromkeys(mutations)

    for res in mutations:
        alpha_scores = []
        beta_scores = []
        angle_scores = [] 

        for i in range(len(d1)):
            alpha_scores.append(bimodal(d1[i], *stats[res]['dist_alpha']))
            beta_scores.append(bimodal(d2[i], *stats[res]['dist_beta']))
            angle_scores.append(bimodal(angle[i], *stats[res]['PAB_angle']))

        alpha_scores = np.array(alpha_scores)
        beta_scores = np.array(beta_scores)
        angle_scores = np.array(angle_scores)

        possible_coordinators = []
        ind_coordinators = np.where((alpha_scores>0.001)&(beta_scores>0.001)&(angle_scores>0.01))[0]
        if len(ind_coordinators)==0:
            fitness = 0
        else:
            fitness = 0
            for ind in ind_coordinators:
                score_ca = alpha_scores[ind]
                score_cb = beta_scores[ind]
                score_angle = angle_scores[ind]
                score_total = (score_ca + score_cb + score_angle) / 3
                possible_coordinators.append(residues_ids[ind])
                fitness += score_total*stats[res]['fitness']
        dic_mutations_scores[res] = [possible_coordinators, fitness]
    return  dic_mutations_scores


def clustering_mutation(scores, dic_coordinating, mutations):

    for coord, score in scores:
        for res_mut, coord_fit in score.items():
            item = {res:0 for res in mutations}
            if len(coord_fit[0]) ==0:
                continue
            else:
                residues_t = tuple(coord_fit[0])
                if residues_t not in dic_coordinating:
                    dic_coordinating[residues_t] = {'probes': [coord], 'score': item}
                    dic_coordinating[residues_t]['score'][res_mut] += coord_fit[1]

                else:
                    dic_coordinating[residues_t]['probes'].append(coord)
                    dic_coordinating[residues_t]['score'][res_mut] += coord_fit[1]

    return dic_coordinating

def geometry(v1: np.array, v2: np.array):
    v1_distances = np.linalg.norm(v1, axis=1)
    v2_distances = np.linalg.norm(v2, axis=1)
    v1_v2_distances = np.linalg.norm(v1 - v2, axis=1)
    if np.any(v1_v2_distances[:] == 0.0):
        return None, None, None
    v1v2_angles = np.arccos(
        (
            np.square(v1_distances) + np.square(v1_v2_distances) -
            np.square(v2_distances)
        ) /
        (2 * v1_distances * v1_v2_distances)
    )
    return v1_distances, v2_distances, v1v2_angles

def bimodal(
    x: np.array,
    chi1: float,
    nu1: float,
    sigma1: float,
    chi2: float,
    nu2: float,
    sigma2: float
) -> np.array or float:
    """
    Helper function for the `gaussian_score` function that computes
    the score associated to a certain set of parameters for the input
    `x`.

    Args:
        x (np.array): Set of input values to evaluate.
        prop (float): Factor describing the contribution of each of
            the gaussians to the final result.
        chi1 (float): Height of the first gaussian.
        nu1 (float): Average value for the first gaussian.
        sigma1 (float): Standard deviation of the first gaussian.
        chi2 (float): Height of the second gaussian.
        nu2 (float): Average value for the second gaussian.
        sigma2 (float): Standard deviation of the second gaussian.

    Returns:
        result (np.array or float): Value or array of values with len(x)
            with the associated probability.
    """
    first_gaussian = _normpdf(x, chi1, nu1, sigma1)
    second_gaussian = _normpdf(x, chi2, nu2, sigma2)
    return first_gaussian + second_gaussian


def _normpdf(x: np.array or float, chi: float, nu: float, std: float):
    """
    Helper function for `double_gaussian`, computes the PDF of a
    gaussian function.

    Args:
        x (np.array or float): Set of input values to evaluate.
        chi (float): Height of the gaussian.
        nu (float): Average value for the gaussian.
        std (float): Standard deviation of the gaussian.

    Returns:
        result (float): PDF value.
    """
    var = std ** 2
    num = np.exp(- (x - nu) ** 2 / (2 * var))
    return chi * num

def centroid(coord_residues):
    for res, probes_score in coord_residues.items():
        probes = np.array(probes_score['probes']).reshape(len(probes_score['probes']),3)
        lenght_array = len(probes)
        sum_x = np.sum(probes[:,0])
        sum_y = np.sum(probes[:,1])
        sum_z = np.sum(probes[:,2])
        centroid = [sum_x/lenght_array, sum_y/lenght_array, sum_z/lenght_array]
        coord_residues[res]['centroid'] = centroid
    return coord_residues    

def new_probes(coord_residues_centroid, alphas, betas, stats, res_number_coordinators):
    for res, values in coord_residues_centroid.items():
        centroid=np.array(values['centroid'])
        new_probes = grid(centroid, 3, 0.6)
        if 'new_scores' not in locals():
            new_scores = coordination_score(alphas, betas, stats, new_probes, res_number_coordinators)
        else:
            new_scores.extend(coordination_score(alphas, betas, stats, new_probes, res_number_coordinators))
    coord_residues_new = clustering(new_scores, coord_residues_centroid)
    return coord_residues_new

def residue_scoring(residues, stats_res):
    score_res = 0
    prop_residues = analyse_residues(residues)
    for group, value in prop_residues.items():
        score_res += bimodal(value, *stats_res[group])
    return score_res

def analyse_residues(residues):
    site = {
            'aromatic': 0,
            'polar':  0,
            'positive': 0,
            'negative': 0,
            'hidrophobic': 0,
            'large_chain': 0
        }
    total = len(residues)
    for res in residues:
        for group, included_residues in chemical_nature.items():
            if res in included_residues:
                site[group] += round(1/total, 2)
    return site


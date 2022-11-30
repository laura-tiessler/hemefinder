import numpy as np


def coordination_score(
    alphas: dict,
    betas: dict,
    stats: dict,
    probes: np.array
) -> np.array:
    score = np.zeros((len(probes), 4))
    score[:, :3] = probes[:, :]

    for idx, probe in enumerate(probes):
        possible_coordinators = 0
        probe_score = 0
        for res in alphas.keys():
            alpha_dists = probe - alphas[res]
            beta_dists = probe - betas[res]
            d1, d2, angle = geometry(alpha_dists, beta_dists)
            probe_score += gaussian_scoring(d1, d2, angle, res, stats)

        score[idx, 3] = probe_score if probe_score < 1.0 else 1.0
    return score


def gaussian_scoring(
    d1: np.array,
    d2: np.array,
    angle: np.array,
    res: str,
    stats: dict
) -> float:
    fitness = 0
    possible_coordinators = 0

    alpha_scores = bimodal(d1, *stats[res]['dist_alpha'])
    beta_scores = bimodal(d2, *stats[res]['dist_beta'])
    angle_scores = bimodal(angle, *stats[res]['PAB_angle'])

    alpha_trues = np.argwhere(alpha_scores > 0.01)
    beta_trues = np.argwhere(beta_scores > 0.01)
    angle_trues = np.argwhere(angle_scores > 0.01)

    for true in alpha_trues:
        if true in beta_trues and true in angle_trues:
            score_1 = alpha_scores[true]
            score_2 = beta_scores[true]
            score_angle = angle_scores[true]
            fitness += (score_1 + score_2 + score_angle) / 3


    return fitness * stats[res]['fitness']


def geometry(v1: np.array, v2: np.array) -> (np.array, np.array, np.array):
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

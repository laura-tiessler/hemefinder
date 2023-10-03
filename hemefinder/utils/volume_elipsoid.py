import os
from math import sqrt
from re import L
import numpy as np
import sys

import pyKVFinder
from numpy import argsort, array, dot, float, identity, linalg, outer, zeros
from sklearn.cluster import KMeans
from scipy import sparse
from scipy.spatial import cKDTree as KDTree

from .parser import load_cav
from .print import print_clusters
from .additional import grid


def normalize(density_points):
    max_p = max(density_points)
    min_p = min(density_points)
    diff = max_p - min_p
    normalized = [float((x - min_p) / diff) for x in density_points]


def detect_hole(cluster_cav_HA, cav):
    """
    This function fills up the holes that are left from metals due to proximity of coordinants. The idea is to use the external points from cavities (HA)
    and find the probes that are surrounded the most by other probes. This are usually regions where there is an opening or a canal.
    This function first calculates a distance matrix with all

    Input:
        - pdb of protein
        - All of values for KVFinder

    Output:
        - list_output: returns a list with the index of the cavities that
        fullfill the requirements
        - it also exports a pdb with these cavities
    """

    cav_lst = list(cav)
    for e, cluster in enumerate(cluster_cav_HA):
        tree = KDTree(cluster)
        density_points = []
        dist_matrix = tree.sparse_distance_matrix(tree, 3.5, p=2.0)

        for a in dist_matrix.toarray():
            close = np.where(a > 0)[0]
            density_points.append(len(close))
        density_points_array = np.array(density_points)
        ind = np.argsort(density_points_array)[-20:]
        coordes = cluster[ind]
        ind_del = []
        for i in range(len(coordes)):
            for j in range(i + 1, len(coordes)):
                dist = np.linalg.norm(coordes[i] - coordes[j])
                if dist < 3:
                    ind_del.append(i)
                    break
        lst_coord = []
        new_coordes = [b for a, b in enumerate(coordes) if a not in ind_del]
        for a, coord in enumerate(new_coordes):
            new_probes = grid(coord, 3, 0.6)
            lst_coord.append(new_probes)
        coord_np = np.array([subitem for item in lst_coord for subitem in item])
        coord_np_no_clashes = delete_close_probes(coord_np, cav)
        for probe in coord_np_no_clashes:
            cav_lst.append(probe)
    cav_more_probes = np.array(cav_lst)
    return cav_more_probes


def delete_close_probes(new_probes, cav):
    # Delete new probes that are very close to other new proves
    ind_to_keep = []
    dist_matrix_cav_other_points = np.sqrt(
        (np.square(new_probes[np.newaxis, :] - new_probes[:, np.newaxis]).sum(axis=2))
    )
    index = [a for a in range(len(new_probes))]
    upper_diagonal_distance_matrix = np.triu(dist_matrix_cav_other_points, k=1)
    for row in upper_diagonal_distance_matrix:
        lst = []
        for i, dist in enumerate(row):
            if dist > 0.0001 and dist < 0.5:
                if i in index:
                    index.remove(i)
    new_probes = np.array(new_probes[index])

    # Delete probes that are very close to original cavity probes
    ind_to_keep = []
    tree1 = KDTree(new_probes)
    tree2 = KDTree(cav)
    density_points = []
    dist_matrix_cav_new_points = tree1.sparse_distance_matrix(tree2, 0.5, p=2.0)
    for f, rows in enumerate(dist_matrix_cav_new_points.toarray()):
        if len(np.where(rows > 0)[0]) == 0:
            ind_to_keep.append(f)
    final_new_probes = new_probes[ind_to_keep]
    return final_new_probes


def kmeans(output_cav, num_clus):
    cav, cav_HA = load_cav(output_cav)

    # Perform Kmeans on CAV_HA to look for points of high density (holes of metal) and build a sphere to fill them
    model = KMeans(
        n_clusters=num_clus,
        init="random",
        n_init=10,
        max_iter=200,
        tol=1e-04,
        random_state=0,
    )
    model.fit(cav_HA)
    kmeans_clu = model.fit_predict(cav_HA)
    cluster_cav_HA = []
    for cluster in range(num_clus):
        clu = np.array((cav_HA[kmeans_clu == cluster]))
        cluster_cav_HA.append(clu)
    new_cav = detect_hole(cluster_cav_HA, cav)

    # Perform Kmeans on probes including filled holes
    model = KMeans(
        n_clusters=num_clus,
        init="random",
        n_init=10,
        max_iter=200,
        tol=1e-04,
        random_state=0,
    )
    model.fit(new_cav)
    kmeans_clu = model.fit_predict(new_cav)
    probes = []
    for cluster in range(num_clus):
        clu = np.array((new_cav[kmeans_clu == cluster]))
        probes.append(clu)
    return probes


def volume_pyKVFinder(atomic, pdb: str, outputdir: str):
    """
    This function calculates the cavities inside the protein using KVFinder.
    It also calculates the volume, surface and area.

    Input:
        - pdb of protein
        - All of values for KVFinder

    Output:
        - list_output: returns a list with the index of the cavities that
        fullfill the requirements
        - it also exports a pdb with these cavities
    """
    tmp_dir = "tmp"
    # Define values for calculation
    step = 0.6
    probe_in = 1.5
    probe_out = 11.0
    removal_distance = 2.5
    volume_cutoff = 5.0
    surface = "SES"
    index = 2

    # Create the 3D grid
    vertices = pyKVFinder.get_vertices(atomic, probe_out=probe_out, step=step)
    ncav, cavities = pyKVFinder.detect(
        atomic,
        vertices,
        step=step,
        probe_in=probe_in,
        probe_out=probe_out,
        removal_distance=removal_distance,
        volume_cutoff=volume_cutoff,
        surface=surface,
    )
    surface, volume, area = pyKVFinder.spatial(cavities, step=step)
    probes = []
    for vol in volume.values():
        vol = float(vol)
        if vol > 130:
            output_cavity = os.path.join(outputdir, f"cavity_{pdb}_{index}.pdb")
            pyKVFinder.export(
                output_cavity, cavities, surface, vertices, selection=[index]
            )
            if vol < 1089:
                cav, cav_HA = load_cav(output_cavity)
                more_probes = detect_hole([cav_HA], cav)
                probes.append(more_probes)
            else:
                num_clus = int(vol / 609)
                probes_clu = kmeans(output_cavity, num_clus)
                probes.extend(probes_clu)
        index += 1
    for a in range(len(probes)):
        output_cluster = os.path.join(outputdir, f"cluster_{pdb}_{a}.xyz")
        with open(output_cluster, "w") as f:
            for coord in probes[a]:
                f.write("He %s %s %s \n" % (coord[0], coord[1], coord[2]))
        f.close()
    message = f"Processed protein {pdb} and found: {len(probes)}"
    message += " pockets with adequate volume."
    print(message)
    return probes


def detect_ellipsoid(probes, center):
    distances = np.sqrt((np.square(probes[:, np.newaxis] - center).sum(axis=2)))
    close = np.where(distances < 7.63)[0]
    sphere = probes[close]
    return sphere


def atoms_inertia(xyz):
    weights = [1 for a in range(len(xyz))]  # fer numpy array de 1
    return [(xyz, weights)]


def moments_of_inertia(vw):
    i = zeros((3, 3), float)
    c = zeros((3,), float)
    w = 0
    for xyz, weights in vw:
        n = len(xyz)
        xyz, weights = array(xyz), array(weights)
        if n > 0:
            wxyz = weights.reshape((n, 1)) * xyz
            w += weights.sum()
            i += (xyz * wxyz).sum() * identity(3) - dot(xyz.transpose(), wxyz)
            c += wxyz.sum(axis=0)
    i /= w
    c /= w  # Center of vertices
    i -= dot(c, c) * identity(3) - outer(c, c)

    eval, evect = linalg.eigh(i)

    # Sort by eigenvalue size.
    order = argsort(eval)
    seval = eval[order]
    sevect = evect[:, order]

    # Make rows of 3 by 3 matrix the principle axes.
    return sevect.transpose(), seval, c


def inertia_ellipsoid_size(d2, shell=False):
    d2sum = sum(d2)
    elen = [sqrt(5 * max(0, (0.5 * d2sum - d2[a]))) for a in range(3)]
    return elen


def elipsoid(sphere):
    """
    This function calculates the elipsoid size of the cavity.

    Input:
        - pdb id to load the cavity
        - list of the cavities indexes to run the calculation
        - path of files

    Output:
        - list_output: returns a list with the index of the cavities that
        fullfill the requirements
        - it also exports a pdb with these cavities
    """
    vw = atoms_inertia(sphere)
    axes, d2, center = moments_of_inertia(vw)
    elen = inertia_ellipsoid_size(d2)
    if elen[0] > 5.51 and elen[1] > 4.74 and elen[2] > 1.82:
        return axes, d2, center, elen
    else:
        return 0, 0, 0, 0


def detect_residues(points, alphas, betas, residue_names):
    dic_residues = {}
    alpha_distances = np.sqrt((np.square(points[:, np.newaxis] - alphas).sum(axis=2)))
    beta_distances = np.sqrt((np.square(points[:, np.newaxis] - betas).sum(axis=2)))
    close_residues = np.unique(
        np.where((alpha_distances < 6.5) & (beta_distances < 5))[1]
    )

    list_residues_site = residue_names[close_residues]

    return list_residues_site

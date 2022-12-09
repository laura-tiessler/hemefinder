import os
from math import sqrt
import numpy as np

import pyKVFinder
from numpy import argsort, array, dot, float, identity, linalg, outer, zeros
from sklearn.cluster import KMeans

from .parser import load_cav
from .print import print_clusters


def kmeans(pdb_id, f, num_clus, outputdir):
    file_cav, xyz_cav = load_cav(pdb_id, f, outputdir)
    model = KMeans(n_clusters=num_clus)
    model.fit(xyz_cav)
    k_means_clusters = model.fit_predict(xyz_cav)
    return k_means_clusters, xyz_cav


def volume_pyKVFinder(
    atomic,
    pdb: str,
    outputdir: str
):
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

    # Define values for calculation
    step = 0.6
    probe_in = 1.5
    probe_out = 10.0
    removal_distance = 2.5
    volume_cutoff = 5.0
    surface = 'SES'
    index = 2

    # Create the 3D grid
    vertices = pyKVFinder.get_vertices(atomic, probe_out=probe_out, step=step)
    ncav, cavities = pyKVFinder.detect(
        atomic, vertices, step=step, probe_in=probe_in, probe_out=probe_out,
        removal_distance=removal_distance, volume_cutoff=volume_cutoff,
        surface=surface
    )
    surface, volume, area = pyKVFinder.spatial(cavities, step=step)
    index_cavities = [i+2 for i, vol in enumerate(volume.values()) if vol >238]
    output_cavity = os.path.join(outputdir, f'cavity_{pdb}.pdb')
    pyKVFinder.export(
        output_cavity, cavities, surface, vertices, selection=index_cavities
    )
    message = f'Processed protein {pdb} and found: {len(index_cavities)}' 
    message += ' pockets with adequate volume.'
    return load_cav(output_cavity)


def atoms_inertia(xyz):
    weights = [1 for a in range(len(xyz))]  # fer numpy array de 1
    return([(xyz, weights)])


def detect_ellipsoid(probes, center):
    distances = np.sqrt((np.square(probes[:,np.newaxis]-center).sum(axis=2)))
    close = np.where(distances<10)[0]
    sphere = probes[close]
    return sphere

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
            i += (xyz*wxyz).sum()*identity(3) - dot(xyz.transpose(), wxyz)
            c += wxyz.sum(axis=0)

    i /= w
    c /= w                         # Center of vertices
    i -= dot(c, c)*identity(3) - outer(c, c)

    eval, evect = linalg.eigh(i)

    # Sort by eigenvalue size.
    order = argsort(eval)
    seval = eval[order]
    sevect = evect[:, order]

    # Make rows of 3 by 3 matrix the principle axes.
    return sevect.transpose(), seval, c


def inertia_ellipsoid_size(d2, shell=False):
    d2sum = sum(d2)
    elen = [sqrt(5*max(0, (0.5*d2sum - d2[a]))) for a in range(3)]
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
    if elen[0] > 6.73 and elen[1] > 4.97 and elen[2] > 2.16:
        return True
    else:
        return False


def detect_residues(points, alphas, betas, residue_names):
    dic_residues = {}
    alpha_distances = np.sqrt((np.square(points[:,np.newaxis]-alphas).sum(axis=2)))
    beta_distances = np.sqrt((np.square(points[:,np.newaxis]-betas).sum(axis=2)))
    close_residues = np.unique(np.where((alpha_distances<6.5)&(beta_distances<5))[1])
    list_residues_site = residue_names[close_residues]

    return list_residues_site
    


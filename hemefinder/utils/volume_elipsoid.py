import os
from math import sqrt
import numpy as np

import pyKVFinder
from numpy import argsort, array, dot, float, identity, linalg, outer, zeros
from sklearn.cluster import KMeans

from .parser import load_cav, load_kmeans
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
    probe_in = 1.0
    probe_out = 7.0
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
    pyKVFinder.export(output_cavity, cavities, surface, vertices, selection=index_cavities)    
    message = f'Processed protein {pdb} and found: {len(index_cavities)} pockets with adequate volume '
    print(message)
    return output_cavity


def atoms_inertia(xyz):
    weights = [1 for a in range(len(xyz))]  # fer numpy array de 1
    return([(xyz, weights)])


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


def elipsoid(pdb_id, dic_out, outputdir):
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
    cavities = []
    list_elip = []
    print('\n\n')

    for ind, ind_k in dic_out.items():
        for i_k in ind_k:
            file_cav, xyz_cav = load_kmeans(pdb_id, ind, i_k, outputdir)
            # file_cav, xyz_cav, traj_cav = load_cav(pdb_id,f,outputdir)
            vw = atoms_inertia(xyz_cav)
            axes, d2, center = moments_of_inertia(vw)
            elen = inertia_ellipsoid_size(d2)
            print(elen)

            if elen[0] > 6.73 and elen[1] > 4.97 and elen[2] > 2.16:
                message = f'From PDB {pdb_id} - Site number {ind}_{i_k} '
                message += 'the elipsoid would fit'
                print(message)
                list_elip.append(ind)
                cavities.append(xyz_cav)
    print()
    return list_elip, cavities


def detect_residues(xyz, alphas, betas, res_for_column, name_for_res):
    dic_residues = {}
    alpha_distances = np.sqrt((np.square(xyz[:,np.newaxis]-alphas).sum(axis=2)))
    beta_distances = np.sqrt((np.square(xyz[:,np.newaxis]-betas).sum(axis=2)))
    close_residues = np.unique(np.where((alpha_distances<6.5)&(beta_distances<5))[1])
    
    #To count GLY in case of backbone consideration
    glycines = [res for res in res_for_column if name_for_res[res_for_column[res]]=='GLY']
    alpha_gly= np.unique(np.where(alpha_distances<4)[1])
    for r in alpha_gly:
        if r in glycines:
            res = name_for_res[res_for_column[r]]
            dic_residues[res] += 1
    
    list_residues_site = []
    #Keep only the count of each residue in backbone
    for a in close_residues:
        res = name_for_res[res_for_column[a]]
        dic_residues[res] += 1
    return dic_residues
    


import os
from math import sqrt
from re import L
import numpy as np

import pyKVFinder
from numpy import argsort, array, dot, float, identity, linalg, outer, zeros
from sklearn.cluster import KMeans

from .parser import load_cav
from .print import print_clusters
from .additional import grid


def detect_hole(cav_HA, cav):
    dist_matrix = np.sqrt((np.square(cav_HA[np.newaxis,:]-cav_HA[:,np.newaxis]).sum(axis=2)))
    density_points = []
    for a in dist_matrix:
        close = np.where(a<4)[0]
        density_points.append(len(close))
    max_p = max(density_points)
    min_p = min(density_points)
    diff = max_p-min_p
    normalized = [float((x-min_p)/diff) for x in density_points]
    nor = np.array(normalized)
    ind_5 = np.argsort(nor)[-20:]
    coordes_5 = cav_HA[ind_5]
    ind_del = []
    for i in range(len(coordes_5)):
        for j in range(i+1, len(coordes_5)):
            dist = np.linalg.norm(coordes_5[i]-coordes_5[j])
            if dist < 3:
                ind_del.append(i)
                break
    lst_coord = []
    new_coordes_5 = [b for a, b in enumerate(coordes_5) if a not in ind_del]
    for coord in new_coordes_5:
        new_probes = grid(coord, 4 , 0.6)
        lst_coord.append(new_probes)
    coord_np = np.array([subitem for item in lst_coord for subitem in item])
    coord_np_no_clashes = delete_close_probes(coord_np,cav)
    return coord_np_no_clashes

def delete_close_probes(new_probes, cav):
    #Delete coord that are very close to original points
    ind_to_keep = []
    dist_matrix_cav_new_points = np.sqrt((np.square(cav[np.newaxis,:]-new_probes[:,np.newaxis]).sum(axis=2)))
    for e, row in enumerate(dist_matrix_cav_new_points):
        if np.any(row<0.8) == False:
            ind_to_keep.append(e)

    final_new_probes = new_probes[ind_to_keep]
    return final_new_probes

def kmeans(output_cav, num_clus):
    cav, cav_HA = load_cav(output_cav)
    more_probes = detect_hole(cav_HA,cav)
    new_cav = np.vstack((cav, more_probes))
    model = KMeans(n_clusters=num_clus, init='random', n_init=10, max_iter=200, tol=1e-04, random_state=0)
    model.fit(new_cav)
    kmeans_clu = model.fit_predict(new_cav)
    probes = []
    for cluster in range(num_clus):
        clu = np.array((new_cav[kmeans_clu==cluster]))
        probes.append(clu)
    return probes

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
    tmp_dir = 'tmp'
    # Define values for calculation
    step = 0.6
    probe_in = 1.5
    probe_out = 11.0
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
    probes = []
    for vol in volume.values():
        vol = float(vol)
        if vol > 238:
            output_cavity = os.path.join(outputdir, f'cavity_{pdb}_{index}.pdb')
            pyKVFinder.export(output_cavity, cavities, surface, vertices, selection=[index]) #Delete later
            if vol <2500:
                cav, cav_HA = load_cav(output_cavity)
                more_probes = detect_hole(cav_HA, cav)
                new = np.vstack((cav, more_probes))
                probes.append(new)                
            else:
                num_clus = int(vol/1000)
                probes_clu = kmeans(output_cavity, num_clus)
                probes.extend(probes_clu)
        index +=1
    for a in range(len(probes)):
        output_cluster = os.path.join(outputdir, f'cluster_{pdb}_{a}.xyz')
        with open(output_cluster, 'w') as f:
            for coord in probes[a]:
                f.write('He %s %s %s \n' %(coord[0], coord[1], coord[2]))
        f.close()
    message = f'Processed protein {pdb} and found: {len(probes)}' 
    message += ' pockets with adequate volume.'
    return probes


def atoms_inertia(xyz):
    weights = [1 for a in range(len(xyz))]  # fer numpy array de 1
    return([(xyz, weights)])


def detect_ellipsoid(probes, center):
    distances = np.sqrt((np.square(probes[:,np.newaxis]-center).sum(axis=2)))
    close = np.where(distances<9)[0]
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
    if elen[0] > 6.27 and elen[1] > 5.33 and elen[2] > 2.08:
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
    


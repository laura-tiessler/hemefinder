from numpy import  zeros, float, array, dot, outer, argsort, linalg, identity
import pyKVFinder



def volume_pyKVFinder(pdb, pdb_id,path_files):

    """
    This function calculates the cavities inside the protein using KVFinder. It also calculates the volume, surface and area.

    Input:
        - pdb of protein  
        - All of values for KVFinder

    Output:
        - list_output: returns a list with the index of the cavities that fullfill the requirements
        - it also exports a pdb with these cavities

    """
    #Read the coordinates of the protein
    atomic = pyKVFinder.read_pdb(pdb)

    #Define values for calculation
    step = 0.6
    probe_in = 1.0
    probe_out = 7.0
    removal_distance = 2.5
    volume_cutoff = 5.0
    surface = 'SES'
    index = 2

    #Create the 3D grid
    vertices = pyKVFinder.get_vertices(atomic, probe_out=probe_out, step=step)
    ncav, cavities = pyKVFinder.detect(atomic, vertices, step=step, probe_in=probe_in, probe_out=probe_out, removal_distance=removal_distance, volume_cutoff=volume_cutoff, surface=surface)
    surface, volume, area = pyKVFinder.spatial(cavities, step=step)
    dic_output = {}
    for k,v in volume.items():
        if float(v)>259:
            output_cavity = 'cavity_' + str(index) + '_' + pdb_id
            pyKVFinder.export(output_cavity, cavities, surface, vertices, selection = [index]) 
            if float(v)>259 and float(v)<2000:
                num_clus = int(1)
                labels, xyz = kmeans(input, index,num_clus, path_files)
                index_kmeans = print_clusters(labels, xyz, 'output_cavities_%s_%s' %(input[:-4],index))
                dic_output[index] = index_kmeans
                print('From pdb id %s site %s: volume is %s' %(pdb_id,index, v))
            elif float(v)>2000:
                num_clus = int(v/1000)
                labels, xyz = kmeans(input, index,num_clus, path_files)
                index_kmeans = print_clusters(labels, xyz, 'output_cavities_%s_%s' %(input[:-4],index))
                dic_output[index] = index_kmeans
        index+= 1
    print('Processed protein %s and found %s pockets with adequate volume' %(pdb_id, len(dic_output))) 
    return dic_output


def atoms_inertia(xyz):
    weights = [1 for a in range(len(xyz))] #fer numpy array de 1
    return([(xyz,weights)])


def moments_of_inertia(vw):
    i = zeros((3,3),float)
    c = zeros((3,),float)
    w = 0
    for xyz, weights in vw:
        n = len(xyz)
        xyz, weights = array(xyz), array(weights)
        if n > 0 :
            wxyz = weights.reshape((n,1)) * xyz
            w += weights.sum()
            i += (xyz*wxyz).sum()*identity(3) - dot(xyz.transpose(),wxyz)
            c += wxyz.sum(axis = 0)

    i /= w
    c /= w                         # Center of vertices
    i -= dot(c,c)*identity(3) - outer(c,c)

    eval, evect = linalg.eigh(i)

    # Sort by eigenvalue size.
    order = argsort(eval)
    seval = eval[order]
    sevect = evect[:,order]

    # Make rows of 3 by 3 matrix the principle axes.
    return sevect.transpose(), seval, c

def inertia_ellipsoid_size(d2, shell = False):
    d2sum = sum(d2)
    from math import sqrt
    elen = [sqrt(5*max(0,(0.5*d2sum - d2[a]))) for a in range(3)]
    return elen


def elipsoid(pdb_id, dic_out,path_files):

    """
    This function calculates the elipsoid size of the cavity.

    Input:
        - pdb id to load the cavity
        - list of the cavities indexes to run the calculation
        - path of files

    Output:
        - list_output: returns a list with the index of the cavities that fullfill the requirements
        - it also exports a pdb with these cavities
    """
    cavities = []
    list_elip = []
    for ind, ind_k in dic_out.items():
        for i_k in ind_k:
            file_cav, xyz_cav = load_kmeans(pdb_id, ind, i_k, path_files)
            #file_cav, xyz_cav, traj_cav = load_cav(pdb_id,f,path_files)
            vw = atoms_inertia(xyz_cav)
            axes, d2, center = moments_of_inertia(vw)
            elen = inertia_ellipsoid_size(d2)
            if elen[0] > 6.73 and elen[1]>4.97 and elen[2]>2.16:
                print('From pdb id %s site number %s %sthe elipsoid would fit' %(pdb_id, str(ind), str(i_k)))
                list_elip.append(ind)
                cavities.append(xyz_cav)
    if list_elip == []:
        to_revise.append([pdb_id, 'elipsoid'])
    return list_elip,cavities

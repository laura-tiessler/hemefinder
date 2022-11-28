import os
import sys
import urllib.request
from urllib.error import HTTPError

import numpy as np
import pyKVFinder

from .data import CONVERT_RES_NAMES


def read_pdb(file, outputdir):

    """
    This fuction loads the pdb file of the protein with MD traf.

    Input:
        - pdb_id: Path to the pdb file id of the pdb

    Output:
        - A numpy array with atomic data (residue number, chain, residue name,
        atom name, xyz coordinates and radius) for each atom.
    """
    if file.endswith('.pdb'):
        atomic = pyKVFinder.read_pdb(file)
    else:
        pdb_file = download_pdb(file, datadir=outputdir)
        print(f'Downloading: {file}')
        atomic = pyKVFinder.read_pdb(str(pdb_file))

    return atomic


def download_pdb(file, datadir):
    """_summary_

    Args:
        file (str): pdb id 
        datadir (str): path where the pdb will be downloaded

    Returns:
        output_filename (str): path of the pdb file downloaded
    """
    pdb_filename = file + '.pdb' 
    url = 'https://files.rcsb.org/download/' + pdb_filename
    outfilename = os.path.join(datadir, pdb_filename)
    try:
        urllib.request.urlretrieve(url, outfilename)
        return outfilename
    except HTTPError as err:
        print(str(err), file=sys.stderr)
        return None


def load_cav(pdb_id, index, outputdir):
    """
    This function load the pdb with the cavities that fullfill the
    requirements.

    Input:
        - pdb id: pdb that you want to load the cavities
        - index: list of indexes of the cavities
    """
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'vdw_mod.dat')
    pdb_cav = os.path.join(outputdir, f'cavity_{index}_{pdb_id}.pdb')
    vdw = pyKVFinder.read_vdw(data_path)
    atomic = pyKVFinder.read_pdb(pdb_cav, vdw)
    xyz = atomic[:, [4, 5, 6]]
    return pdb_cav, xyz


def load_kmeans(pdb_id, index,kmean_index, path_files):

    """
    This function load the pdb with the cavities that fullfill the requirements.

    Input:
        - pdb id: pdb that you want to load the cavities
        - index: list of indexes of the cavities 

    
    """
    pdb_cav = 'output_cavities_' + str(pdb_id[:-4]) + '_' + str(index) + '_' + str(kmean_index) + '.xyz'
    file = os.path.join(path_files,pdb_cav)
    xyz=[]
    with open(file, 'r') as f:
        for line in f:
            coord = [float(a) for a in line.replace("\n","").split(' ')[1:4]]
            xyz.append(coord)
    xyz =np.array(xyz)
    return file, xyz


def _parse_molecule(lines, file_extension):
    """
    Parses a protein and generates the necessary data for the calculation.
    All the atoms of the protein areparsed by searching those of type alpha
    carbon, beta carbon, backbone carbon, backbone nitrogen and backbone oxygen.
    Their coordinates are stored in separated numpy arrays for further use in
    the BioMetAll calculation. Also, it generates dictionaries to control the
    correspondence between residues and column numbers of the numpy arrays.
    Finally, the centroid and distance to the furthest atom of the protein are
    calculated to allow the generation of the grid of probes.
    Parameters
    ----------
    lines : array_like
        Array of str containing the structure of the protein
    file_extension : str
        Extension indicating the format of the `lines`
    Returns
    -------
    np.array
        3-float numpy array containing the centroid of the protein
    float
        Distance to the furthest atom, adding a security margin to account for
        a possible superficial coordination
    np.array
        Array of 3-D coordinates for all the alpha carbons of the protein
    np.array
        Array of 3-D coordinates for all the beta carbons of the protein
    np.array
        Array of 3-D coordinates for all the backbone carbons of the protein
    np.array
        Array of 3-D coordinates for all the backbone nitrogens of the protein
    np.array
        Array of 3-D coordinates for all the backbone oxygens of the protein
    np.array
        Array of 3-D coordinates for all the side-chain atoms of the protein
    dict
        Correspondence between number:chain of residue and column number
    dict
        Correspondence between column number and number of residue:chain
    dict
        Name of the residue given its number:chain
    dict
        Name of atoms contained in a given residue (indexed by number_res:chain)
    """
    if file_extension != '.pdb':
        return None
    # Extract residue information and assign column
    i = 0
    column_for_res = {}
    res_for_column = {}
    name_for_res = {}
    atoms_in_res = {}
    for line in lines:
        record_type = line[0:6]
        if record_type == "ATOM  ":
            atom_fullname = line[12:16]
            # get rid of whitespace in atom names
            split_list = atom_fullname.split()
            if len(split_list) != 1:
                # atom name has internal spaces, e.g. " N B ", so
                # we do not strip spaces
                atom_name = atom_fullname
            else:
                # atom name is like " CA ", so we can strip spaces
                atom_name = split_list[0]

            if atom_name in ['CA', 'CB', 'C', 'N', 'O']:
                altloc = line[16]
                chainid = line[21]
                resid = line[22:26].split()[0]
                res = str(resid) + ":" + str(chainid)
                resname = line[17:20]
                if resname in list(CONVERT_RES_NAMES):
                    resname = CONVERT_RES_NAMES[resname]
                if res not in list(column_for_res):
                    column_for_res[res] = i
                    res_for_column[i] = res
                    name_for_res[res] = resname
                    atoms_in_res[res] = set()
                    i += 1
                atoms_in_res[res].add(atom_name)

    #Extract coordinates and atoms information
    alphas = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    betas = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    carbons = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    nitrogens = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    oxygens = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    side_chains = []
    coords_array = [] #For calculate grid size

    for line in lines:
        record_type = line[0:6]
        if record_type == "ATOM  ":
            atom_fullname = line[12:16]
            # get rid of whitespace in atom names
            split_list = atom_fullname.split()
            if len(split_list) != 1:
                # atom name has internal spaces, e.g. " N B ", so
                # we do not strip spaces
                atom_name = atom_fullname
            else:
                # atom name is like " CA ", so we can strip spaces
                atom_name = split_list[0]

            chainid = line[21]
            resid = line[22:26].split()[0]
            res = str(resid) + ":" + str(chainid)

            # atomic coordinates
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                raise Exception("Invalid or missing coordinate(s) at \
                                residue %s, atom %s" % (res, atom_name))
            coord = [x, y, z]
            if atom_name == "CA":
                # Coordinates for the grid
                coords_array.append(coord)
                # Coordinates for searching sites
                alphas[column_for_res[res]] = coord
            elif atom_name == "CB":
                coords_array.append(coord)
                # Coordinates for searching sites
                betas[column_for_res[res]] = coord
            elif atom_name == "C":
                coords_array.append(coord)
                # Coordinates for searching sites
                carbons[column_for_res[res]] = coord
            elif atom_name == "N":
                coords_array.append(coord)
                # Coordinates for searching sites
                nitrogens[column_for_res[res]] = coord
            elif atom_name == "O":
                coords_array.append(coord)
                # Coordinates for searching sites
                oxygens[column_for_res[res]] = coord
            else: # Atom belongs to a side-chain
                coords_array.append(coord)
                # Coordinates for discarding clashes
                side_chains.append(coord)
    
    coords_array = np.array(coords_array)
    alphas = np.array(alphas)
    betas = np.array(betas)
    carbons = np.array(carbons)
    nitrogens = np.array(nitrogens)
    oxygens = np.array(oxygens)
    side_chains = np.array(side_chains)

    return alphas, betas, carbons, nitrogens, oxygens, column_for_res, res_for_column, name_for_res, atoms_in_res, side_chains, coords_array
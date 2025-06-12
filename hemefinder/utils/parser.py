from http.server import BaseHTTPRequestHandler
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
    if file.endswith(".pdb"):
        atomic = pyKVFinder.read_pdb(file)
    else:
        pdb_file = download_pdb(file, datadir=outputdir)
        print(f"Downloading: {file}")
        atomic = pyKVFinder.read_pdb(str(pdb_file))

    return atomic


def find_coordinators(atomic: np.array, coordinators: list, stats: dict) -> list:
    new_coordinators = []

    if coordinators is not None:
        for coor in coordinators:
            if coor in stats.keys():
                new_coordinators.append(coor)
        return new_coordinators

    possible_coordinators = []
    cummulative = 0.0

    for res_name, res_info in stats.items():
        possible_coordinators.append((res_name, res_info["fitness"]))

    possible_coordinators = sorted(
        possible_coordinators, key=lambda x: x[1], reverse=True
    )

    for coor in possible_coordinators:
        new_coordinators.append(coor[0])
        cummulative += coor[1]
        if cummulative >= 0.95:
            return new_coordinators

    return new_coordinators


def parse_residues(target: str, molecule: np.array, coordinators: list, stats: dict):
    # coordinators = find_coordinators(molecule, coordinators, stats)

    alphas_coord = []
    betas_coord = []
    res_name_number_coord = []
    all_alphas = []
    all_betas = []
    residues_names = []
    residues_ids = []
    for atom in molecule:
        res_name = atom[2]
        res_num = int(atom[0])
        res_chain = atom[1]
        res_id = str(str(res_num) + "_" + res_chain)
        atom_name = atom[3]
        coors = np.array(atom[4:7], dtype=float)

        # For residue calculation
        if atom_name == "CA":
            all_alphas.append(coors)
            residues_names.append(res_name)
            residues_ids.append(res_id)
            if res_name == "GLY":
                all_betas.append(np.array([0, 0, 0]))
        if atom_name == "CB":
            all_betas.append(coors)

        if res_name not in coordinators:
            continue

        if atom_name == "CA":
            alphas_coord.append(coors)
            res_name_number_coord.append([res_name, res_id])
        elif atom_name == "CB":
            betas_coord.append(coors)

    alphas_coord = np.array(alphas_coord)
    betas_coord = np.array(betas_coord)
    res_name_number_coord = np.array(res_name_number_coord)
    all_alphas = np.array(all_alphas)
    all_betas = np.array(all_betas)

    residues_names = np.array(residues_names)
    residues_ids = np.array(residues_ids)

    if len(all_alphas) != len(all_betas):
        raise IndexError("There is an error in residue composition. Some residues might be incomplete, lacking alpha or beta carbons.")

    return (
        alphas_coord,
        betas_coord,
        res_name_number_coord,
        all_alphas,
        all_betas,
        residues_names,
        residues_ids,
    )


def download_pdb(file, datadir):
    """_summary_

    Args:
        file (str): pdb id
        datadir (str): path where the pdb will be downloaded

    Returns:
        output_filename (str): path of the pdb file downloaded
    """
    pdb_filename = f"{file}.pdb"
    url = f"https://files.rcsb.org/download/{pdb_filename}"
    outfilename = os.path.join(datadir, pdb_filename)
    try:
        urllib.request.urlretrieve(url, outfilename)
        return outfilename
    except HTTPError as err:
        print(str(err), file=sys.stderr)
        return None


def load_cav(cavity_path):
    """
    This function load the pdb with the cavities that fullfill the
    requirements.

    Input:
        - pdb id: pdb that you want to load the cavities
        - outputdir:

    Output:
        - xyz_cav: xyz of the cavity
    """
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "vdw_mod.dat")
    vdw = pyKVFinder.read_vdw(data_path)
    atomic = pyKVFinder.read_pdb(cavity_path, vdw)
    xyz_cav = np.array(atomic[:, [4, 5, 6]], dtype=float)
    xyz_HA = np.array([a[4:7] for a in atomic if a[3] == "HA"], dtype=float)
    return xyz_cav, xyz_HA


# def load_cav(cavity_path):
#     """

#     """
#     current_dir = os.path.dirname(__file__)
#     data_path = os.path.join(current_dir, 'vdw_mod.dat')
#     vdw = pyKVFinder.read_vdw(data_path)
#     atomic = pyKVFinder.read_pdb(cavity_path, vdw)
#     xyz_HA = np.array([a[4:7] for a in atomic if a[3]=='HA'], dtype=float )
#     return  xyz_HA


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
    if file_extension != ".pdb":
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

            if atom_name in ["CA", "CB", "C", "N", "O"]:
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

    # Extract coordinates and atoms information
    alphas = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    betas = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    carbons = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    nitrogens = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    oxygens = [[0.0, 0.0, 0.0] for i in range(0, len(list(column_for_res)))]
    side_chains = []
    coords_array = []  # For calculate grid size

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
                raise Exception(
                    "Invalid or missing coordinate(s) at \
                                residue %s, atom %s"
                    % (res, atom_name)
                )
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
            else:  # Atom belongs to a side-chain
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

    return (
        alphas,
        betas,
        carbons,
        nitrogens,
        oxygens,
        column_for_res,
        res_for_column,
        name_for_res,
        atoms_in_res,
        side_chains,
        coords_array,
    )

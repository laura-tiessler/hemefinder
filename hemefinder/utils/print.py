import os

import numpy as np


def print_clusters(cluster_num, xyz, output_file):
    np_cluster= np.unique(cluster_num)
    dict_clusters = {}
    index_kmeans = []

    for pos, cluster in np.ndenumerate(cluster_num):
        if cluster not in dict_clusters:
            dict_clusters[cluster] = list([xyz[pos]])
        else:
            dict_clusters[cluster].extend([xyz[pos]])

    for key in dict_clusters.keys():
        output = output_file + '_' + str(key) + '.xyz'
        with open(output, 'w') as f:
            for coord in dict_clusters[key]:
                f.write('He %s %s %s \n' %(coord[0], coord[1], coord[2]))
        f.close()
        index_kmeans.append(key)
    return index_kmeans


def _print_pdb(sorted_data, filename):
    """
    Generates a .pdb file containing the probes of the BioMetAll calculation.
    Each coordinating environment obtained from the BioMetAll calculation is
    stored as a different residue, being the centroid probe a Helium atom and
    all the probes Xeon atoms.
    Parameters
    ----------
    sorted_data : array_like
        Data obtained from the clustering of BioMetAll results
    filename : str
        Name of the output .pdb file. Usually with format `probes_xxxx.pdb` in
        the working directory.
    """
    file_pdb = open(filename,"w")
    num_at = 0
    num_res = 0
    for one_result in sorted_data:
        chains = set()
        for r in one_result[0]:
            r = r.strip("_BCK")
            chains.add(r.split(":")[1])
        cen_str = ""
        for r in one_result[1]:
            crd_center = "{:.8s}".format(str(round(float(r),3)))
            if len(crd_center)<8:
                crd_center = " "*(8-len(crd_center)) + crd_center
                cen_str += crd_center
            else:
                cen_str += crd_center
        num_at += 1
        num_res += 1
        for ch in chains:
            file_pdb.write("ATOM" +" "*(7-len(str(num_at))) + "%s  HE  SLN %s" %(num_at, ch))
            file_pdb.write(" "*(3-len(str(num_res))) + "%s     %s  1.00  0.00          HE\n" %(num_res, cen_str))
        for prob in one_result[4]:
            num_at += 1
            prb_str = ""
            for p in prob:
                prb_center = "{:.8s}".format(str(round(float(p),3)))
                if len(prb_center)<8:
                    prb_center = " "*(8-len(prb_center)) + prb_center
                    prb_str += prb_center
                else:
                    prb_str += prb_center
            for ch in chains:
                file_pdb.write("ATOM" +" "*(7-len(str(num_at))) + "%s  XE  SLN %s" %(num_at, ch))
                file_pdb.write(" "*(3-len(str(num_res))) + "%s     %s  1.00  0.00          XE\n" %(num_res, prb_str))
    file_pdb.close()

def print_file(centers, motif, propose_mutations_to, mutations, name_for_res, pdb, inputfile, cmd_str, filename = 'output'):
    sorted_data = sorted(centers, key=lambda x: x[2], reverse=True)
    if motif:
        file_name_addendum = "_" + motif.replace("[", "").replace("]", "").replace(",", "_").replace("/", "-")
    else:
        file_name_addendum = ""
    if pdb:
        pdb_filename = "probes_%s%s.pdb" %(os.path.basename(filename), file_name_addendum)
        pdb_filename = os.path.join(os.path.dirname(inputfile), pdb_filename)
        _print_pdb(sorted_data, pdb_filename)

    mutations_width, radius_width, coord_width, probes_width, residues_width, pos_width = len('Proposed mutations'), len('Radius search'), len('Coordinates of center'), len('Num. probes'), len('Coordinating residues'), 1
    lines=[]
    for pos, one_center in enumerate(sorted_data,1):
        residues = []
        for res in one_center[0]:
            if not "_BCK" in res:
                residues.append(str(name_for_res[res]) + ":" + res)
            else:
                res_bck = res.split("_")[0]
                residues.append(str(name_for_res[res_bck]) + ":" + res_bck + "_BCK")
                continue

        residues_str = ' '.join(res for res in residues)
        coord_str = ' '.join([str(format(r, '.3f')) for r in one_center[1]])
        pos_str, radius_str, probes_str = str(pos), str(format(one_center[3], '.3f')), str(one_center[2])

        if propose_mutations_to:
            mutations_str = ' '.join([str(res) + ":" + str(mut) for (res,mut) in mutations[one_center[0]]])
            if len(mutations_str) > mutations_width:
                mutations_width = len(mutations_str)
        else:
            mutations_str = ''
        if len(pos_str) > pos_width:
            pos_width = len(pos_str)
        if len(coord_str) > coord_width:
            coord_width = len(coord_str)
        if len(probes_str) > probes_width:
            probes_width = len(probes_str)
        if len(residues_str) > residues_width:
            residues_width = len(residues_str)
        if len(radius_str) > radius_width:
            radius_width = len(radius_str)
        lines.append((pos_str, residues_str, coord_str, probes_str, radius_str, mutations_str))
    print(' {:>{pos_width}} | {:^{residues_width}} | {:{coord_width}} | {:{probes_width}} | {:{radius_width}} | {:{mutations_width}}'.format(
            '#', 'Coordinating residues', 'Coordinates of center', 'Num. probes', 'Radius search' , 'Proposed mutations', pos_width=pos_width, residues_width=residues_width, coord_width=coord_width,
            probes_width=probes_width, radius_width=radius_width, mutations_width=mutations_width))
    print('-{}-+-{}-+-{}-+-{}-+-{}-+-{}-'.format('-'*pos_width, '-'*residues_width, '-'*coord_width, '-'*probes_width, '-'*radius_width, '-'*mutations_width))
    for line in lines:
        print(' {:>{pos_width}} | {:<{residues_width}} | {:^{coord_width}} | {:>{probes_width}} | {:<{radius_width}} | {:<{mutations_width}} '.format(
                line[0], line[1], line[2], line[3], line[4], line[5], pos_width=pos_width, residues_width=residues_width,
                coord_width=coord_width, probes_width=probes_width, radius_width=radius_width, mutations_width=mutations_width))

    text_filename = "results_biometall_%s%s.txt" %(os.path.basename(filename), file_name_addendum)
    text_filename = os.path.join(os.path.dirname(inputfile), text_filename)
    f = open(text_filename, "w")
    str_header = "*****HemeFinder"
    f.write(str_header)
    f.write('\n')
    f.write(cmd_str)
    f.write('\n')
    f.write(' {:>{pos_width}} | {:^{residues_width}} | {:{coord_width}} | {:{probes_width}} | {:{radius_width}} | {:{mutations_width}} \n'.format(
            '#', 'Coordinating residues', 'Coordinates of center', 'Num. probes', 'Radius search' , 'Proposed mutations', pos_width=pos_width, residues_width=residues_width, coord_width=coord_width,
            probes_width=probes_width, radius_width=radius_width, mutations_width=mutations_width))
    f.write('-{}-+-{}-+-{}-+-{}-+-{}-+-{}-\n'.format('-'*pos_width, '-'*residues_width, '-'*coord_width, '-'*probes_width, '-'*radius_width, '-'*mutations_width ))
    for line in lines:
        f.write(' {:>{pos_width}} | {:<{residues_width}} | {:^{coord_width}} | {:>{probes_width}} | {:<{radius_width}} | {:<{mutations_width}} \n'.format(
                line[0], line[1], line[2], line[3], line[4], line[5], pos_width=pos_width, residues_width=residues_width,
                coord_width=coord_width, probes_width=probes_width, radius_width=radius_width, mutations_width=mutations_width))
    f.close()
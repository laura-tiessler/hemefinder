import json
import os


def load_stats():
    current_dir = os.path.dirname(__file__)
    stats_path = os.path.join(current_dir, 'stats', 'heme_gaus_statistics.json')
    with open(stats_path) as json_reader:
        stats = json.load(json_reader)
    return stats

def load_stats_res():
    current_dir = os.path.dirname(__file__)
    stats_path = os.path.join(current_dir, 'stats', 'residue_stats.json')
    with open(stats_path) as json_reader:
        stats = json.load(json_reader)
    return stats
    
def load_stats_two_coord():
    current_dir = os.path.dirname(__file__)
    stats_path = os.path.join(current_dir, 'stats', 'two_coord_data.json')
    with open(stats_path) as json_reader:
        stats = json.load(json_reader)
    return stats


DIST_PROBE_ALPHA = {
    'CYS': (3.702, 4.561),
    'ASP': (5.617, 5.617),
    'SER': (3.211, 4.795),
    'LYS': (5.095, 9.040),
    'PRO': (2.758, 3.462),
    'HIS': (5.624, 7.111),
    'ARG': (10.148, 10.148),
    'TRP': (0.482, 8.3215),
    'GLU': (4.745, 6.932),
    'TYR': (3.635, 11.185),
    'MET': (4.354, 6.376),
    'ALA': (4.751, 4.751), 
    'ALL': (4.510, 6.858)
}

DIST_PROBE_BETA = {
    'CYS': (3.040, 3.729),
    'ASP': (4.454, 4.454),
    'SER': (2.709, 3.478),
    'LYS': (4.068, 8.314),
    'PRO': (4.130, 4.488),
    'HIS': (5.203, 6.125),
    'ARG': (8.888, 8.888),
    'TRP': (2.397, 7.082),
    'GLU': (4.159, 5.871),
    'TYR': (4.705, 9.045),
    'MET': (3.597, 4.981),
    'ALA': (4.878, 4.878), 
    'ALL': (4.352, 5.944)
}

ANGLE_PAB = {
    'CYS': (0.663, 1.114),
    'ASP': (0.609, 0.609),
    'SER': (0.384, 1.142),
    'LYS': (-0.406, 2.015),
    'PRO': (2.135, 2.526),
    'HIS': (0.541, 1.417),
    'ARG': (0.557, 0.557),
    'TRP': (0.651, 2.638),
    'GLU': (0.550, 1.212),
    'TYR': (-0.174, 2.412),
    'MET': (-0.029, 1.338),
    'ALA': (1.493, 1.493), 
    'ALL': (0.581, 1.539)
}



CONVERT_RES_NAMES = {
    'CYX': 'CYS',
    'CYM': 'CYS',
    'ASH': 'ASN',
    'GLH': 'GLN',
    'HIE': 'HIS',
    'HID': 'HIS',
    'HIP': 'HIS',
    'HYP': 'PRO',
    'LYN': 'LYS'
}


RESIDUE_LIST = ['CYS', 'ASP', 'SER', 'GLN', 'LYS',
               'ILE', 'PRO', 'THR', 'PHE', 'ASN',
               'GLY', 'HIS', 'LEU', 'ARG', 'TRP',
               'ALA', 'VAL', 'GLU', 'TYR', 'MET']

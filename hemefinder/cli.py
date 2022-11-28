"""Console script for hemefinder."""
import argparse
import os
import sys


def client() -> dict:
    """Console script for hemefinder."""
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str,
                        help='Molecule PDB file to be analysed.')
    parser.add_argument('--outputdir', type=str, default='.',
                        help='Directory where outputs should be stored.')
    args = vars(parser.parse_args())

    # Prepare output directory
    if not os.path.isdir(args['outputdir']):
        os.mkdir(args['outputdir'])
    return args


if __name__ == "__main__":
    sys.exit(client())  # pragma: no cover

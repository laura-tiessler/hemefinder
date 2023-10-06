"""Console script for hemefinder."""
import argparse
import os
import sys


def client() -> dict:
    """Console script for hemefinder."""
    p = argparse.ArgumentParser()
    p.add_argument("target", type=str, help="Molecule PDB file to be analysed.")
    p.add_argument(
        "--outputdir",
        type=str,
        default=".",
        help="Directory where outputs should be stored.",
    )
    p.add_argument(
        "--coordinators",
        type=str,
        default='["HIS","TYR","CYS","MET",]',
        help="List of possible coordinating residues.",
    )
    p.add_argument(
        "--mutations",
        type=list,
        default=[],
        help="List of possible mutating residues.",
    )
    args = vars(p.parse_args())

    # Prepare output directory
    if not os.path.isdir(args["outputdir"]):
        os.mkdir(args["outputdir"])
    return args


if __name__ == "__main__":
    sys.exit(client())  # pragma: no cover

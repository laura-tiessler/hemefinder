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
        default='[HIS,TYR,CYS,MET]',
        help="List of possible coordinating residues.",
    )
    p.add_argument(
        "--min_num_coordinators",
        type=int,
        default=1,
        help="Minum number of coordinants.",
    )
    p.add_argument(
        "--mutations",
        type=list,
        default=[],
        help="List of possible mutating residues.",
    )
    p.add_argument(
        "--probe_in",
        type=float,
        default=1.5,
        help="Probe in pyKVFinder.",
    )
    p.add_argument(
        "--probe_out",
        type=float,
        default=11.0,
        help="Probe out pyKVFinder.",
    )
    p.add_argument(
        "--removal_distance",
        type=float,
        default=2.5,
        help="Removal distance.",
    )
    p.add_argument(
        "--volume_cutoff",
        type=float,
        default=1.5,
        help="Volume cutoff",
    )
    p.add_argument(
        "--surface",
        type=str,
        default="SES",
        help="SES or SASA",
    )
    args = vars(p.parse_args())

    # Prepare output directory
    if not os.path.isdir(args["outputdir"]):
        os.mkdir(args["outputdir"])
    return args


if __name__ == "__main__":
    sys.exit(client())  # pragma: no cover

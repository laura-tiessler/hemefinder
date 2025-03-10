"""Console script for hemefinder."""

import typer
from .hemefinder import hemefinder


def main():
    hemefinder("hemefinder/data/7bc7.pdb", 'output')

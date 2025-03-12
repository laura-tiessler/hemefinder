"""Console script for hemefinder."""

import typer
from .hemefinder import hemefinder


def main():
    typer.run(hemefinder)

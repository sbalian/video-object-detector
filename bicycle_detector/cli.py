import concurrent.futures
import fractions
import functools
import pathlib
from typing import Annotated

import typer

from . import video

app = typer.Typer()


@app.command()
def frames(
    input: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory to search for .dav files in to extract from.",
        ),
    ],
    output: Annotated[
        pathlib.Path,
        typer.Option(
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Output directory.",
        ),
    ] = pathlib.Path("."),
    workers: Annotated[
        int,
        typer.Option(
            help="Number of maximum workers (0 for number of available CPUs).",
        ),
    ] = 0,
):
    """Extract frames from .dav videos in INPUT to JPEGs on disk."""

    output.mkdir(exist_ok=True)
    paths = sorted(list(input.glob("*.dav")))

    extract = functools.partial(
        video.extract_frames, fps=fractions.Fraction(1, 3), output_root=output
    )

    if workers == 0:
        max_workers = None
    else:
        max_workers = workers

    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        list(executor.map(extract, paths))


@app.command()
def classify():
    """Run bicycle classification on JPEGs in INPUT directory."""
    pass
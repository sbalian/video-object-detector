import concurrent.futures
import fractions
import functools
import pathlib
from typing import Annotated, Optional

import typer

from . import classifier, video

app = typer.Typer()


def _frame_extraction_worker(
    video_path: pathlib.Path, output_directory: pathlib.Path
) -> None:
    video_name = video_path.stem
    frames_directory = output_directory / video_name
    frames_directory.mkdir(exist_ok=True)
    video.extract_frames(
        video_path, fractions.Fraction(1, 3), frames_directory
    )


@app.command()
def frames(
    input_directory: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory containing target .dav files.",
        ),
    ],
    output_directory: Annotated[
        pathlib.Path,
        typer.Option(
            "--output-directory",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Output directory.",
        ),
    ] = pathlib.Path("."),
    max_workers: Annotated[
        Optional[int],
        typer.Option(
            "--max-workers",
            "-w",
            help="Number of maximum workers. Uses all CPUs when unset.",
        ),
    ] = None,
):
    """Extract frames from .dav videos in INPUT to JPEGs on disk."""

    output_directory.mkdir(exist_ok=True)
    video_paths = sorted(list(input_directory.glob("*.dav")))

    worker = functools.partial(
        _frame_extraction_worker, output_directory=output_directory
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        list(executor.map(worker, video_paths))


@app.command()
def classify(
    input_directory: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory containing target JPEG files.",
        ),
    ],
):
    classifier.run_for_frames(input_directory)

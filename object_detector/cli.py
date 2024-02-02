import concurrent.futures
import fractions
import functools
import os
import pathlib
from typing import Annotated, Optional

import loguru
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
    """Extract frames from .dav videos in INPUT_DIRECTORY to JPEGs on disk."""

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
            help="Directory containing target frames subdirectories.",
        ),
    ],
):
    """Classify JPEG frames in subdirectories of INPUT_DIRECTORY.

    Writes a pred.json file in each subdirectory containing the predictions.
    """

    subdirectories = []
    for file in os.listdir(input_directory):
        path = input_directory / file
        if path.is_dir():
            subdirectories.append(path)

    for frames_directory in subdirectories:
        classifier.run_for_frames(frames_directory)
        loguru.logger.info(f"Wrote {frames_directory / 'pred.json'}")

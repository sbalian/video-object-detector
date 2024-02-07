import concurrent.futures
import datetime
import fractions
import functools
import pathlib
from typing import Annotated, Optional

import rich.console
import rich.progress
import typer
from loguru import logger

from . import classifier, media

app = typer.Typer()

DATETIME_FORMAT = "%m-%d-%Y %H:%M:%S"


def fraction(value: str) -> fractions.Fraction:
    return fractions.Fraction(value)


@app.command()
def extract_frames(
    input_directory: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory containing videos.",
        ),
    ],
    fps: Annotated[
        fractions.Fraction,
        typer.Option(
            "--fps",
            "-f",
            help="Frames per second to extract. "
            "For example, 25/1 means extract 25 frames every 1 second.",
            parser=fraction,
        ),
    ] = fractions.Fraction(1, 3),
    workers: Annotated[
        Optional[int],
        typer.Option(
            "--workers",
            "-w",
            help="Number of maximum workers passed to multiprocessing (number "
            "of available CPUs if unset).",
        ),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="ffmpeg output.",
        ),
    ] = "%04d.jpeg",
):
    """Extract frames from videos in INPUT_DIRECTORY.

    For each video a <name>.frames/ subdirectory is created containing the
    extracted frames, where <name> is the name of the video without its
    extension.
    """

    console = rich.console.Console()

    worker = functools.partial(
        media.extract_frames,
        fps=fps,
        ffmpeg_output=output,
    )

    with console.status(f"Looking for videos in {input_directory} ..."):
        paths = [
            path
            for path in input_directory.glob("*")
            if media.is_likely_video(path)
        ]
    console.print(f"Found {len(paths)} videos.")

    console.print(
        "Extraction started: "
        f"{datetime.datetime.now().strftime(DATETIME_FORMAT)}"
    )
    with console.status("Extracting frames ..."):
        with concurrent.futures.ProcessPoolExecutor(workers) as executor:
            futures = [executor.submit(worker, path) for path in paths]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except media.FrameExtractionError as extraction_error:
            logger.error(extraction_error)
    console.print(
        "Extraction ended: "
        f"{datetime.datetime.now().strftime(DATETIME_FORMAT)}"
    )


@app.command()
def detect(
    input_directory: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory containing images.",
        ),
    ],
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size.",
        ),
    ] = 10,
    use_gpu: Annotated[
        bool,
        typer.Option(
            "--gpu/--no-gpu", help="Use the GPU or just use the CPU."
        ),
    ] = True,
):
    """Classify images in INPUT_DIRECTORY.

    Writes the predictions to INPUT_DIRECTORY/predictions.od. Each line in
    the output is a JSON containing the classification result for an image.
    """

    console = rich.console.Console()

    output_path = input_directory / "predictions.od"

    if output_path.exists() and output_path.is_dir():
        logger.error(f"cannot have directory {output_path}")
        raise typer.Exit(code=1)

    with console.status(f"Looking for images in {input_directory} ..."):
        paths = [
            path
            for path in input_directory.glob("*")
            if media.is_likely_image(path)
        ]
    console.print(f"Found {len(paths)} images.")

    console.print(
        "Classification started: "
        f"{datetime.datetime.now().strftime(DATETIME_FORMAT)}"
    )
    with console.status("Classifying ..."):
        try:
            predictions = classifier.batch_run(paths, batch_size, use_gpu)
        except classifier.NoGPUError:
            logger.error("No GPU found. Set --no-gpu.")
            raise typer.Exit(code=1)

    console.print(
        "Classification ended: "
        f"{datetime.datetime.now().strftime(DATETIME_FORMAT)}"
    )

    with open(output_path, "a") as f:
        for prediction in predictions:
            f.write(prediction.model_dump_json())
            f.write("\n")

    console.print(f"Wrote results to {output_path}.")

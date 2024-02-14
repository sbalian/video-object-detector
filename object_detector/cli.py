import datetime
import fractions
import os
import pathlib
from typing import Annotated

import typer
from loguru import logger

from . import classifier, display, media

app = typer.Typer()

DATETIME_FORMAT = "%m-%d-%Y %H:%M:%S"


def fraction(value: str) -> fractions.Fraction:
    return fractions.Fraction(value)


@app.command()
def extract_frames(
    video_path: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Path to input video file.",
        ),
    ],
    output_directory: Annotated[
        pathlib.Path,
        typer.Argument(
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Output directory in which frames will be written.",
        ),
    ],
    fps: Annotated[
        fractions.Fraction,
        typer.Option(
            "--fps",
            "-f",
            help=(
                "Frames per second. For example, 25/1 means extract "
                "25 frames every 1 second."
            ),
            parser=fraction,
        ),
    ] = fractions.Fraction(1, 3),
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            "-o",
            help=(
                "Output format. The ffmpeg output is "
                "output_directory/output_format."
            ),
        ),
    ] = "%04d.jpeg",
):
    """Extract frames from VIDEO_PATH writing into OUTPUT_DIRECTORY."""

    display.PRINT(
        "Extraction started: "
        f"{datetime.datetime.now().strftime(DATETIME_FORMAT)}"
    )
    with display.STATUS("Extracting frames ..."):
        try:
            media.extract_frames(
                video_path, fps, output_directory, output_format
            )
        except media.FrameExtractionError as extraction_error:
            logger.error(extraction_error)
        else:
            display.PRINT(
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
    output_path: Annotated[
        pathlib.Path,
        typer.Option(
            "--output-path",
            "-o",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Output path.",
        ),
    ] = pathlib.Path("predictions.jsonl"),
    force_cpu: Annotated[
        bool,
        typer.Option(
            "--force-cpu/--no-force-cpu",
            help="Use the CPU even if a GPU is available.",
        ),
    ] = False,
):
    """Classify the images found by recursively searching INPUT_DIRECTORY.

    Each line in the output is a JSON containing the classification result
    for an image.
    """

    with display.STATUS(f"Looking for images in {input_directory} ..."):
        image_paths = []
        for root, _, files in os.walk(input_directory):
            for file in files:
                if media.is_likely_image(pathlib.Path(file)):
                    image_paths.append(pathlib.Path(root) / file)

    display.PRINT(
        "Classification started: "
        f"{datetime.datetime.now().strftime(DATETIME_FORMAT)}"
    )

    classifier.batch_run(image_paths, batch_size, force_cpu, output_path)

    display.PRINT(
        "Classification ended: "
        f"{datetime.datetime.now().strftime(DATETIME_FORMAT)}"
    )

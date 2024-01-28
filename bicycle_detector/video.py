import fractions
import pathlib

import ffmpeg


def extract_frames(
    video_path: pathlib.Path,
    fps: fractions.Fraction,
    output_directory: pathlib.Path,
) -> None:
    if output_directory.exists() and output_directory.is_file():
        raise NotADirectoryError(f"{output_directory}")
    output_directory.mkdir(exist_ok=True)
    path = output_directory / "%04d.jpeg"
    str_fps = f"{fps.numerator}/{fps.denominator}"
    ffmpeg.input(video_path).filter("fps", fps=str_fps).output(
        str(path), loglevel="quiet"
    ).run()
    return

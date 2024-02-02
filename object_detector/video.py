"""Video manipulation using ffmpeg."""

import fractions
import pathlib

import ffmpeg


def is_video(path: pathlib.Path) -> bool:
    """Returns True if `path` has at least one video channel."""

    if not path.exists():
        raise FileNotFoundError(f"{path}")
    if path.is_dir():
        raise IsADirectoryError(f"{path}")

    probe = ffmpeg.probe(path)
    for stream in probe["streams"]:
        if stream["codec_type"] == "video":
            return True

    return False


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

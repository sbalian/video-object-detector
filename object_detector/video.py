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
    ffmpeg_output: str = "%04d.jpeg",
) -> None:
    """Extract frames from `video_path`.

    Frames are written to a new directory `video_path`.frames.

    `fps` sets the frames per second.

    `ffmpeg_output` defaults to '%04d.jpeg'.
    """

    output_directory = video_path.parent / f"{video_path.stem}.frames"

    if output_directory.exists() and output_directory.is_file():
        raise NotADirectoryError(f"{output_directory}")
    output_directory.mkdir(exist_ok=False)

    ffmpeg.input(video_path).filter(
        "fps", fps=f"{fps.numerator}/{fps.denominator}"
    ).output(
        (output_directory / ffmpeg_output).as_posix(), loglevel="quiet"
    ).run()

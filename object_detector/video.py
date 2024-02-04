"""Video manipulation using ffmpeg."""

import fractions
import mimetypes
import pathlib

import ffmpeg

# Video extensions outside of MIME
EXTRA_VIDEO_TYPES = (".dav",)


def is_likely_video(path: pathlib.Path) -> bool:
    """Returns True if `path` is likely a video (uses MIME)."""

    type_, _ = mimetypes.guess_type(path)
    if type_ is None:
        extension = path.suffix
        return extension in EXTRA_VIDEO_TYPES
    else:
        return type_.split("/")[0] == "video"


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
    output_directory.mkdir(exist_ok=True)

    ffmpeg.input(video_path).filter(
        "fps", fps=f"{fps.numerator}/{fps.denominator}"
    ).output(
        (output_directory / ffmpeg_output).as_posix(), loglevel="quiet"
    ).run()

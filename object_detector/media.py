"""Video and image manipulation."""

import fractions
import mimetypes
import pathlib

import ffmpeg

# Video extensions outside of MIME (lowercase and with the .)
EXTRA_VIDEO_TYPES = (".dav",)


class FrameExtractionError(Exception):
    pass


def is_likely_video(path: pathlib.Path) -> bool:
    """Returns True if `path` is likely a video (uses MIME)."""

    if path.is_dir():
        return False

    type_, _ = mimetypes.guess_type(path)
    if type_ is None:
        extension = path.suffix
        return extension.lower() in EXTRA_VIDEO_TYPES
    else:
        return type_.split("/")[0] == "video"


def is_likely_image(path: pathlib.Path) -> bool:
    """Returns True if `path` is likely an image (uses MIME)."""

    if path.is_dir():
        return False

    type_, _ = mimetypes.guess_type(path)
    if type_ is None:
        return False
    else:
        return type_.split("/")[0] == "image"


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

    try:
        ffmpeg.input(video_path).filter(
            "fps", fps=f"{fps.numerator}/{fps.denominator}"
        ).output(
            (output_directory / ffmpeg_output).as_posix(), loglevel="error"
        ).run(
            capture_stderr=True
        )
    except ffmpeg.Error as ffmpeg_error:
        raise FrameExtractionError(ffmpeg_error.stderr.decode())

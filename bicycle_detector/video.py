import fractions
import pathlib

import ffmpeg


def extract_frames(
    video_path: pathlib.Path,
    fps: fractions.Fraction,
    output_root: pathlib.Path,
) -> None:
    output_dir = output_root / video_path.name
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "%04d.jpeg"
    ffmpeg.input(video_path).filter(
        "fps", fps=f"{fps.numerator}/{fps.denominator}"
    ).output(str(path), loglevel="quiet").run()

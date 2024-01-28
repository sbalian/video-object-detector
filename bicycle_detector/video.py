import fractions
import pathlib

import ffmpeg
import numpy as np
from PIL import Image


def extract_frames(
    video_path: pathlib.Path, fps: fractions.Fraction, stream_index: int = 0
) -> list[Image.Image]:
    probe = ffmpeg.probe(video_path)
    stream = probe["streams"][stream_index]
    width = int(stream["width"])
    height = int(stream["height"])
    out, err = (
        ffmpeg.input(video_path)
        .filter("fps", fps=f"{fps.numerator}/{fps.denominator}")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet")
        .run(capture_stdout=True)
    )
    if err is not None:
        raise RuntimeError("ffmpeg error")
    frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return [
        Image.fromarray(frames[i, :, :, :]) for i in range(frames.shape[0])
    ]

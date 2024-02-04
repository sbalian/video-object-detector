import fractions
import os
import pathlib

import moviepy.editor as mpy
import numpy as np
import pytest

from object_detector import media


def make_random_image(time) -> np.ndarray:
    imarray = np.random.rand(100, 100, 3) * 255
    return imarray.astype("uint8")


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "foo.mp4"
    clip = mpy.VideoClip(make_random_image, duration=5)
    clip.write_videofile(path.as_posix(), fps=25)
    return path


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (pathlib.Path("test"), False),
        (pathlib.Path("test.mp3"), False),
        (pathlib.Path("test.jpeg"), False),
        (pathlib.Path("test.txt"), False),
        (pathlib.Path("test.wav"), False),
        (pathlib.Path("test.mp4"), True),
        (pathlib.Path("test.dav"), True),
        (pathlib.Path("test.mov"), True),
        (pathlib.Path("test.wmv"), True),
        (pathlib.Path("test.avi"), True),
    ],
)
def test_is_likely_video(test_input, expected):
    assert media.is_likely_video(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (pathlib.Path("test"), False),
        (pathlib.Path("test.mp3"), False),
        (pathlib.Path("test.jpeg"), True),
        (pathlib.Path("test.txt"), False),
        (pathlib.Path("test.wav"), False),
        (pathlib.Path("test.mp4"), False),
        (pathlib.Path("test.png"), True),
        (pathlib.Path("test.svg"), True),
    ],
)
def test_is_likely_image(test_input, expected):
    assert media.is_likely_image(test_input) == expected


def test_extract_frames(sample_video):
    media.extract_frames(sample_video, fractions.Fraction(5, 1))
    assert sorted(
        os.listdir(sample_video.parent / f"{sample_video.stem}.frames")
    ) == [f"{i + 1}".zfill(4) + ".jpeg" for i in range(25)]


def test_extract_frames_raises_for_text_file(tmp_path):
    path = tmp_path / "file.txt"
    with open(path, "w") as f:
        f.write("foo")
    with pytest.raises(media.FrameExtractionError):
        media.extract_frames(path, fractions.Fraction(5, 1))

import fractions
import os
import pathlib

import pytest

from object_detector import video


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
    assert video.is_likely_video(test_input) == expected


def test_extract_frames(sample_video):
    video.extract_frames(sample_video, fractions.Fraction(5, 1))
    assert sorted(
        os.listdir(sample_video.parent / f"{sample_video.stem}.frames")
    ) == [f"{i + 1}".zfill(4) + ".jpeg" for i in range(25)]

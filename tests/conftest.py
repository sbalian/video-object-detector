import pathlib

import moviepy.editor as mpy
import numpy as np
import pytest
import wavio


def make_random_image(time: float) -> np.ndarray:
    imarray = np.random.rand(100, 100, 3) * 255
    return imarray.astype("uint8")


def make_random_video(path: pathlib.Path) -> None:
    clip = mpy.VideoClip(make_random_image, duration=2)
    clip.write_videofile(path.as_posix(), fps=30)


def make_random_wav(path: pathlib.Path) -> None:
    wavio.write(path.as_posix(), np.ones(44100), 44100, sampwidth=3)


@pytest.fixture
def sample_video(tmp_path):
    make_random_video(tmp_path / "foo.mp4")
    return tmp_path / "foo.mp4"


@pytest.fixture
def sample_wav(tmp_path):
    make_random_wav(tmp_path / "foo.wav")
    return tmp_path / "foo.wav"

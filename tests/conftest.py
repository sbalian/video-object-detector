import moviepy.editor as mpy
import numpy as np
import pytest


def make_random_image(time) -> np.ndarray:
    imarray = np.random.rand(100, 100, 3) * 255
    return imarray.astype("uint8")


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "foo.mp4"
    clip = mpy.VideoClip(make_random_image, duration=5)
    clip.write_videofile(path.as_posix(), fps=25)
    return path

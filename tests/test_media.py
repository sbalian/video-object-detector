import fractions
import os
import pathlib
import shutil

import pytest

from object_detector import media


@pytest.fixture
def sample_video(tmp_path):
    shutil.copy("tests/data/cat.mp4", tmp_path)
    return tmp_path / "cat.mp4"


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (pathlib.Path("test"), False),
        (pathlib.Path("test.mp3"), False),
        (pathlib.Path("test.jpeg"), True),
        (pathlib.Path("test.jpg"), True),
        (pathlib.Path("test.JPG"), True),
        (pathlib.Path("test.txt"), False),
        (pathlib.Path("test.wav"), False),
        (pathlib.Path("test.mp4"), False),
        (pathlib.Path("test.png"), True),
        (pathlib.Path("test.svg"), True),
        (pathlib.Path("test.heif"), True),
    ],
)
def test_is_likely_image(test_input, expected):
    assert media.is_likely_image(test_input) == expected


def test_is_likely_image_returns_false_for_dir(tmp_path):
    test_dir = tmp_path / "test.jpeg"
    test_dir.mkdir()
    assert not media.is_likely_image(test_dir)


def test_extract_frames(sample_video, tmp_path):
    frames_directory = tmp_path / "frames"
    media.extract_frames(
        sample_video, fractions.Fraction(1, 2), frames_directory
    )
    assert sorted(os.listdir(frames_directory)) == [
        f"{i + 1}".zfill(4) + ".jpeg" for i in range(8)
    ]


def test_extract_frames_raises_for_text_file(tmp_path):
    frames_directory = tmp_path / "frames"
    path = tmp_path / "file.txt"
    with open(path, "w") as f:
        f.write("foo")
    with pytest.raises(media.FrameExtractionError):
        media.extract_frames(path, fractions.Fraction(5, 1), frames_directory)

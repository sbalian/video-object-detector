import fractions
import os
import pathlib

import pytest

from object_detector import video


def test_is_video_returns_false_for_directory(tmp_path):
    directory = tmp_path / "foo"
    directory.mkdir()
    assert not video.is_video(directory)


def test_is_video_raises_for_non_existing_file():
    with pytest.raises(FileNotFoundError):
        video.is_video(pathlib.Path("foo"))


def test_is_video_returns_false_for_non_media_file(tmp_path):
    path = tmp_path / "foo.txt"
    with open(path, "w") as f:
        f.write("foo")
    assert not video.is_video(path)


def test_is_video_returns_false_for_wav_file(sample_wav):
    assert not video.is_video(sample_wav)


def test_is_video_returns_true_for_video_file(sample_video):
    assert video.is_video(sample_video)


def test_extract_frames(sample_video):
    video.extract_frames(sample_video, fractions.Fraction(5, 1))
    assert sorted(
        os.listdir(sample_video.parent / f"{sample_video.stem}.frames")
    ) == [f"{i + 1}".zfill(4) + ".jpeg" for i in range(25)]

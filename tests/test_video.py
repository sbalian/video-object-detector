import pathlib

import ffmpeg
import pytest

from object_detector import video


def test_is_video_raises_for_directory(tmp_path):
    directory = tmp_path / "foo"
    directory.mkdir()
    with pytest.raises(IsADirectoryError):
        video.is_video(directory)


def test_is_video_raises_for_non_existing_file():
    with pytest.raises(FileNotFoundError):
        video.is_video(pathlib.Path("foo"))


def test_is_video_raises_for_non_media_file(tmp_path):
    with open(tmp_path / "foo.txt", "w") as f:
        f.write("foo")
    with pytest.raises(ffmpeg._run.Error):
        video.is_video(tmp_path / "foo.txt")


def test_is_video_returns_false_for_wav_file(sample_wav):
    assert not video.is_video(sample_wav)


def test_is_video_returns_true_for_video_file(sample_video):
    assert video.is_video(sample_video)

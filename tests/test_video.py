import fractions
import os
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


def test_extract_frames(sample_video):
    video.extract_frames(sample_video, fractions.Fraction(5, 1))
    assert sorted(
        os.listdir(sample_video.parent / f"{sample_video.stem}.frames")
    ) == [f"{i + 1}".zfill(4) + ".jpeg" for i in range(25)]


def test_extract_frames_raises_for_existing_directory(sample_video):
    (sample_video.parent / f"{sample_video.stem}.frames").mkdir()
    with pytest.raises(FileExistsError):
        video.extract_frames(sample_video, fractions.Fraction(5, 1))


def test_extract_frames_raises_for_existing_file(sample_video):
    with open(sample_video.parent / f"{sample_video.stem}.frames", "w") as f:
        f.write("foo")
    with pytest.raises(NotADirectoryError):
        video.extract_frames(sample_video, fractions.Fraction(5, 1))

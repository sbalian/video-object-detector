import datetime
import pathlib

import pytest

from object_detector import classifier


@pytest.fixture
def sample_prediction():
    return classifier.Prediction(
        labels=["car", "cat", "bicycle"],
        scores=[0.1, 0.9, 0.2],
        image_path=pathlib.Path("image.jpg"),
        timestamp=datetime.datetime(2024, 2, 4, 23, 25, 53, 112918),
    )


@pytest.fixture
def sample_prediction_json():
    return dict(
        labels=["car", "cat", "bicycle"],
        scores=[0.1, 0.9, 0.2],
        image_path="image.jpg",
        timestamp=1707089153.112918,
    )


def test_prediction_top_n(sample_prediction):
    assert sample_prediction.top_n(2, 0.05) == [(0.9, "cat"), (0.2, "bicycle")]
    assert sample_prediction.top_n(2, 0.5) == [(0.9, "cat")]
    assert sample_prediction.top_n(1, 0.8) == [(0.9, "cat")]


def test_prediction_to_json(sample_prediction, sample_prediction_json):
    assert sample_prediction.to_json() == sample_prediction_json


def test_prediction_from_json(sample_prediction, sample_prediction_json):
    assert (
        classifier.Prediction.from_json(sample_prediction_json)
        == sample_prediction
    )


def test_classifier_predicts():
    # Cat from https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/360px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg  # noqa

    clf = classifier.Classifier(use_gpu=False)
    [prediction1, prediction2] = clf.predict(
        [
            pathlib.Path("tests/data/cat.jpg"),
            pathlib.Path("tests/data/cat.jpg"),
        ]
    )
    for prediction in [prediction1, prediction2]:
        _, label = prediction.top_n(1, 0.98)[0]
        assert label == "cat"


def test_batch_run(mocker):
    prediction = classifier.Prediction(
        labels=["cat"],
        scores=[0.9],
        image_path=pathlib.Path("tests/data/cat.jpg"),
        timestamp=datetime.datetime.now(),
    )
    batch_size = 10
    total_length = 100
    mocker.patch(
        "object_detector.classifier.Classifier.predict",
        return_value=[prediction] * batch_size,
    )
    assert (
        classifier.batch_run(
            [pathlib.Path("tests/data/cat.jpg")] * total_length,
            batch_size=batch_size,
        )
        == [prediction] * total_length
    )

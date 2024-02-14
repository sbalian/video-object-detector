import datetime
import json
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


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((2, 0.05), [(0.9, "cat"), (0.2, "bicycle")]),
        ((2, 0.5), [(0.9, "cat")]),
        ((1, 0.8), [(0.9, "cat")]),
    ],
)
def test_prediction_top_classes(test_input, expected, sample_prediction):
    assert sample_prediction.top_classes(*test_input) == expected


def test_prediction_check_threshold():
    assert classifier.Prediction.check_threshold(0.5) is None
    with pytest.raises(classifier.ThresholdError):
        assert classifier.Prediction.check_threshold(1.1) is None
    with pytest.raises(classifier.ThresholdError):
        assert classifier.Prediction.check_threshold(-2) is None


def test_prediction_contains(sample_prediction):
    assert sample_prediction.contains("cat", 0.89)
    assert not sample_prediction.contains("cat", 0.95)


def test_classifier_predicts():
    clf = classifier.Classifier(force_cpu=True)
    [prediction1, prediction2] = clf.predict(
        [
            pathlib.Path("tests/data/cat.jpg"),
            pathlib.Path("tests/data/cat.jpg"),
        ]
    )
    for prediction in [prediction1, prediction2]:
        _, label = prediction.top_classes(1, 0.98)[0]
        assert label == "cat"


def test_batch_run(mocker, tmp_path):
    prediction = classifier.Prediction(
        labels=["cat"],
        scores=[0.9],
        image_path=pathlib.Path("cat.jpg"),
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
            [pathlib.Path("cat.jpg")] * total_length,
            batch_size,
            True,
            tmp_path / "predictions.jsonl",
        )
        == [prediction] * total_length
    )
    predictions_from_file = [
        classifier.Prediction(**json.loads(data))
        for data in pathlib.Path(tmp_path / "predictions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert predictions_from_file == [prediction] * total_length

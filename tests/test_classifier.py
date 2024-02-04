import pathlib

from object_detector import classifier


def test_prediction_top_n():
    prediction = classifier.Prediction(
        labels=["car", "cat", "bicycle"],
        scores=[0.1, 0.9, 0.2],
        image_path=pathlib.Path("image.jpg"),
    )
    assert prediction.top_n(2, 0.05) == [(0.9, "cat"), (0.2, "bicycle")]
    assert prediction.top_n(2, 0.5) == [(0.9, "cat")]
    assert prediction.top_n(1, 0.8) == [(0.9, "cat")]


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

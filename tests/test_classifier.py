import pathlib

from PIL import Image

from object_detector import classifier


def test_prediction_top_n():
    prediction = classifier.Prediction(
        labels=["car", "cat", "bicycle"], scores=[0.1, 0.9, 0.2]
    )
    assert prediction.top_n(2, 0.05) == [(0.9, "cat"), (0.2, "bicycle")]
    assert prediction.top_n(2, 0.5) == [(0.9, "cat")]
    assert prediction.top_n(1, 0.8) == [(0.9, "cat")]


def test_classifier_predicts():
    # Cats from ...
    # Cat 1: https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/360px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg  # noqa
    # Cat 2: https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/640px-Cat_August_2010-4.jpg  # noqa

    images = [
        Image.open(path)
        for path in ["tests/data/cat1.jpg", "tests/data/cat2.jpg"]
    ]
    clf = classifier.Classifier()
    [prediction1, prediction2] = clf.predict(images)
    for prediction in [prediction1, prediction2]:
        _, label = prediction.top_n(1, 0.98)[0]
        assert label == "cat"


def test_batch_run(mocker):
    mocker.patch(
        "object_detector.classifier.Classifier.predict",
        return_value=[
            classifier.Prediction(labels=["car", "cat"], scores=[0.1, 0.9])
        ]
        * 2,
    )
    image_paths = [
        pathlib.Path(path)
        for path in ["tests/data/cat1.jpg", "tests/data/cat2.jpg"] * 10
    ]
    classifier.batch_run(image_paths, batch_size=2)

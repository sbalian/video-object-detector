"""Object detection using https://huggingface.co/facebook/detr-resnet-101."""

import datetime
import pathlib
import warnings

import pydantic
import torch
import transformers
from PIL import Image

from . import display


class GPUNotFoundError(Exception):
    pass


class ThresholdError(ValueError):
    pass


class Prediction(pydantic.BaseModel):
    labels: list[str]
    scores: list[float]
    image_path: pathlib.Path
    timestamp: datetime.datetime

    @staticmethod
    def check_threshold(threshold: float) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ThresholdError(
                f"threshold {threshold} out of bounds [0.0, 1.0]"
            )

    def contains(self, class_: str, threshold: float) -> bool:
        """Check if `class_` has score > `threshold`."""

        self.check_threshold(threshold)

        return class_ in set(
            label
            for score, label in zip(self.scores, self.labels)
            if score > threshold
        )

    def top_classes(self, n: int, threshold: float) -> list[tuple[float, str]]:
        """Return top `n` classes sorted by score > `threshold`."""

        self.check_threshold(threshold)

        return sorted(
            [
                (score, label)
                for score, label in zip(self.scores, self.labels)
                if score > threshold
            ],
            reverse=True,
            key=lambda x: x[0],
        )[:n]


class Classifier:
    def __init__(self, force_cpu: bool) -> None:
        self.force_cpu = force_cpu
        model_name = "facebook/detr-resnet-101"
        revision = "no_timm"
        self.processor = transformers.DetrImageProcessor.from_pretrained(
            model_name, revision=revision
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="torch._utils",
                lineno=831,
            )
            self.model = transformers.DetrForObjectDetection.from_pretrained(
                model_name, revision=revision
            )
        if not self.force_cpu and torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, image_paths: list[pathlib.Path]) -> list[Prediction]:
        """Return predictions for a list of images `images_paths`."""

        images = [Image.open(image_path) for image_path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        if not self.force_cpu and torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="torch.nn.modules.conv",
                lineno=456,
            )
            outputs = self.model(**inputs)
        out_logits = outputs.logits
        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        return [
            Prediction(
                labels=[
                    self.model.config.id2label[label.item()]
                    for label in labels_
                ],
                scores=[float(score) for score in scores_],
                image_path=image_path,
                timestamp=datetime.datetime.now(),
            )
            for scores_, labels_, image_path in zip(
                scores, labels, image_paths
            )
        ]


def batch_run(
    image_paths: list[pathlib.Path],
    batch_size: int,
    force_cpu: bool,
    output_path: pathlib.Path,
) -> list[Prediction]:
    """Classify list of images `image_paths` in batches sized `batch_size`."""

    clf = Classifier(force_cpu=force_cpu)
    predictions = []

    with display.PROGRESS:
        for i in display.PROGRESS.track(
            range(0, len(image_paths), batch_size),
            description="Classifying ...",
        ):
            batch = image_paths[i : i + batch_size]
            batch_predictions = clf.predict(batch)
            with open(output_path, "a") as f:
                for prediction in batch_predictions:
                    f.write(prediction.model_dump_json())
                    f.write("\n")
            predictions.extend(batch_predictions)
    return predictions

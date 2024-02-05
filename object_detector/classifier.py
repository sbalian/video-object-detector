"""Detect objects in images using facebook/detr-resnet-101.

Model from: https://huggingface.co/facebook/detr-resnet-101.
"""

import dataclasses
import datetime
import pathlib

import pydantic
import torch
import transformers
from PIL import Image


class NoGPUError(Exception):
    pass


class Prediction(pydantic.BaseModel):
    labels: list[str]
    scores: list[float]
    image_path: pathlib.Path
    timestamp: datetime.datetime

    def top_n(self, n: int, threshold: float) -> list[tuple[float, str]]:
        """Return top `n` labels sorted by score > `threshold`."""

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold {threshold} out of bounds [0.0, 1.0]")
        return sorted(
            [
                (score, label)
                for score, label in zip(self.scores, self.labels)
                if score > threshold
            ],
            reverse=True,
            key=lambda x: x[0],
        )[:n]


@dataclasses.dataclass
class Classifier:
    processor: transformers.DetrImageProcessor = dataclasses.field(init=False)
    model: transformers.DetrForObjectDetection = dataclasses.field(init=False)
    use_gpu: bool = True

    def __post_init__(self):
        model_name = "facebook/detr-resnet-101"
        revision = "no_timm"
        self.processor = transformers.DetrImageProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.model = transformers.DetrForObjectDetection.from_pretrained(
            model_name, revision=revision
        )
        if self.use_gpu:
            if torch.cuda.is_available():
                self.model.to("cuda")
            else:
                raise NoGPUError

    def predict(self, image_paths: list[pathlib.Path]) -> list[Prediction]:
        """Return predictions for a list of `images_paths`."""

        images = [Image.open(image_path) for image_path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        if self.use_gpu:
            inputs = inputs.to("cuda")

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
    image_paths: list[pathlib.Path], batch_size: int
) -> list[Prediction]:
    """Classify `image_paths` in batches of size `batch_size`."""

    clf = Classifier()
    predictions = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        predictions.extend(clf.predict(batch))
    return predictions

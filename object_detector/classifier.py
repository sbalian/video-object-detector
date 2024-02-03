"""Detect objects in images using facebook/detr-resnet-101.

Model from: https://huggingface.co/facebook/detr-resnet-101.
"""

import dataclasses
import pathlib

import torch
import transformers
from PIL import Image


@dataclasses.dataclass
class Prediction:
    labels: list[str]
    scores: list[float]

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

    def __post_init__(self):
        model_name = "facebook/detr-resnet-101"
        revision = "no_timm"
        self.processor = transformers.DetrImageProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.model = transformers.DetrForObjectDetection.from_pretrained(
            model_name, revision=revision
        )
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, images: list[Image.Image]) -> list[Prediction]:
        """Return predictions for a list of PIL images `images`.

        Uses GPU if CUDA is available.
        """

        inputs = self.processor(images=images, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        outputs = self.model(**inputs)

        out_logits = outputs.logits

        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        results: list[Prediction] = []
        for scores_, labels_ in zip(scores, labels):
            prediction = Prediction(
                labels=[
                    self.model.config.id2label[label.item()]
                    for label in labels_
                ],
                scores=[float(score) for score in scores_],
            )
            results.append(prediction)
        return results


def batch_run(
    image_paths: list[pathlib.Path], batch_size: int = 10
) -> list[Prediction]:

    clf = Classifier()
    predictions = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(image_path) for image_path in batch]
        predictions.extend(clf.predict(images))
    return predictions

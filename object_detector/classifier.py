import dataclasses
import json
import pathlib

import torch
import transformers
from PIL import Image


@dataclasses.dataclass
class Prediction:
    labels: list[str]
    scores: list[float]


@dataclasses.dataclass
class Classifier:
    processor: transformers.DetrImageProcessor = dataclasses.field(init=False)
    model: transformers.DetrForObjectDetection = dataclasses.field(init=False)

    def __post_init__(self):
        self.processor = transformers.DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-101", revision="no_timm"
        )
        self.model = transformers.DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101", revision="no_timm"
        )
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, images: list[Image.Image]) -> list[Prediction]:
        inputs = self.processor(images=images, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        outputs = self.model(**inputs)

        out_logits = outputs.logits

        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        results = []
        for score, label in zip(scores, labels):
            prediction = Prediction(
                labels=[self.model.config.id2label[ll.item()] for ll in label],
                scores=[float(s) for s in score],
            )
            results.append(prediction)
        return results


def run_for_frames(frames_directory: pathlib.Path) -> None:
    image_paths = list(frames_directory.glob("*.jpeg"))

    clf = Classifier()
    batch_size = 10

    results = {}
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(image_path) for image_path in batch]
        predictions = clf.predict(images)
        for image_path, prediction in zip(batch, predictions):
            results[str(image_path)] = {
                "scores": prediction.scores,
                "labels": prediction.labels,
            }

    with open(frames_directory / "pred.json", "w") as f:
        json.dump(results, f)

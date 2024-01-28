import dataclasses
import fractions
import pathlib

import loguru
import torch
import transformers
from PIL import Image

from . import video


def chunks(list_: list, n: int):
    for i in range(0, len(list_), n):
        yield list_[i : i + n]


@dataclasses.dataclass
class Classifier:
    processor: transformers.DetrImageProcessor = dataclasses.field(init=False)
    model: transformers.DetrForObjectDetection = dataclasses.field(init=False)
    threshold: float = 0.9

    def __post_init__(self):
        self.processor = transformers.DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-101", revision="no_timm"
        )
        self.model = transformers.DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101", revision="no_timm"
        )
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, images: list[Image.Image]) -> list[bool]:
        inputs = self.processor(images=images, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        outputs = self.model(**inputs)

        out_logits = outputs.logits

        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        has_bicycle = []
        for score, label in zip(scores, labels):
            if "car" in [
                self.model.config.id2label[ll.item()]
                for ll in label[score > self.threshold]
            ]:
                has_bicycle.append(True)
            else:
                has_bicycle.append(False)

        return has_bicycle


def run_for_video(
    video_path: pathlib.Path, output_directory: pathlib.Path
) -> None:
    loguru.logger.info(f"Processing {video_path}")

    if output_directory.is_file():
        raise NotADirectoryError(f"{output_directory}")
    if not output_directory.exists():
        output_directory.mkdir()
    video_output_directory = output_directory / video_path.name
    video_output_directory.mkdir(exist_ok=True)
    extracted_frames = video.extract_frames(
        video_path=video_path, fps=fractions.Fraction(1, 3)
    )

    num_extracted_frames = len(extracted_frames)
    batch_size = 10

    clf = Classifier()

    results = []

    for i in range(0, num_extracted_frames, batch_size):
        frames = extracted_frames[i : i + batch_size]
        results.extend(clf.predict(frames))

    counter = 0
    for result, frame in zip(results, extracted_frames):
        if result:
            save_path = video_output_directory / f"{counter}.jpg"
            frame.save(save_path)
            counter += 1

    return

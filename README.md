# Detect objects of interest in videos

I set this up to try and find a stolen bike in hours of CCTV footage. It
uses [detr-resnet-101](https://huggingface.co/facebook/detr-resnet-101)
by
[Carion et al., End-to-End Object Detection with Transformers, 2020](https://arxiv.org/abs/2005.12872)
to locate objects in videos.

## Installation

Install [Poetry](https://python-poetry.org/). Then:

```bash
sudo apt update
sudo apt install ffmpeg --yes

git clone <repo-path>
cd <repo-path>
poetry install
```

## Usage

There are two steps: frame extraction and frame classification.

```bash
poetry run object-detect extract-frames videos/
```

This will extract frames from the videos found in `videos/`.

```bash
poetry run object-detect detect videos/video1.frames/
```

This will classify the frames for `video1`, writing the results to
`videos/video1.frames/predictions.od` (prediction JSON for image per line).

For more CLI options:

```bash
poetry run object-detect --help
poetry run object-detect extract-frames --help
poetry run object-detect detect --help
```

## Development

Follow the installation instructions above. To run the tests:

```bash
./test.sh
```

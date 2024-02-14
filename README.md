# Detect objects of interest in videos

I set this up to try and find a stolen bike in hours of CCTV footage. It
uses [detr-resnet-101](https://huggingface.co/facebook/detr-resnet-101)
by
[Carion et al., End-to-End Object Detection with Transformers, 2020](https://arxiv.org/abs/2005.12872)
to locate objects in videos.

## Installation

Make sure you have Python 3.10+. Next, install
[Poetry](https://python-poetry.org/). Then:

```bash
# Install ffmpeg. For example, on Linux:
sudo apt update
sudo apt install ffmpeg --yes

# Clone this repo and navigate to it
git clone <repo-path>
cd <repo-path>

# Install the Python package, its dependencies and the
# object-detect CLI tool
poetry install
```

## Usage

There are two steps: frame extraction and object detection in the extracted
frames.

```bash
poetry run object-detect extract-frames myvideo.mp4 frames/
```

This will extract frames from `myvideo.mp4` into `frames/`.

```bash
poetry run object-detect detect frames/
```

This will classify all the frames in `frames/`. The prediction
results will be written to `predictions.jsonl`.
Each line in a JSONL file will contain the prediction result for a
single frame.

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

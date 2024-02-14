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

There are two steps: frame extraction and image classification.

```bash
poetry run object-detect extract-frames myvideo.mp4 frames/
```

This will extract frames from `myvideo.mp4` into `frames/`. By default,
the frames are saved as JPEGs.

```bash
poetry run object-detect detect frames/
```

This will classify all the JPEGs in `frames/`. The prediction
results are written to `predictions.jsonl`.
Each line in this JSONL file contains the prediction result for a
single frame.

If you would like to print the paths of images containing a certain class:

```bash
poetry run object-detect show predictions.jsonl cat
```

For more CLI options:

```bash
poetry run object-detect --help
poetry run object-detect extract-frames --help
poetry run object-detect detect --help
poetry run object-detect show --help
```

**Tip**: To run the extraction in parallel for multiple videos,
use `xargs` with `-P`. For example:

```bash
find "/path/to/videos" -name "*.mp4" | xargs -P $(nproc) -I % poetry run object-detect extract-frames % %.frames
```

## Development

Follow the installation instructions above. To run the tests:

```bash
./test.sh
```

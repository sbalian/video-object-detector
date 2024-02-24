# Detect objects in videos

This can be useful if you're looking for something of interest in
videos. For example, a stolen bike in hours of CCTV footage. It
uses [detr-resnet-101](https://huggingface.co/facebook/detr-resnet-101)
by
[Carion et al., End-to-End Object Detection with Transformers, 2020](https://arxiv.org/abs/2005.12872).

Full citation:

```
@article{DBLP:journals/corr/abs-2005-12872,
  author    = {Nicolas Carion and
               Francisco Massa and
               Gabriel Synnaeve and
               Nicolas Usunier and
               Alexander Kirillov and
               Sergey Zagoruyko},
  title     = {End-to-End Object Detection with Transformers},
  journal   = {CoRR},
  volume    = {abs/2005.12872},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.12872},
  archivePrefix = {arXiv},
  eprint    = {2005.12872},
  timestamp = {Thu, 28 May 2020 17:38:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2005-12872.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## Installation

Install [ffmpeg](https://ffmpeg.org/). For example, on Linux with `apt`:

```bash
sudo apt update
sudo apt install ffmpeg --yes
```

Make sure you have Python 3.10.
Next, install [Poetry](https://python-poetry.org/). Then:

```bash
git clone https://github.com/sbalian/video-object-detector.git
cd video-object-detector/
poetry install
```

This installs the Python package, its dependencies and the
`object-detect` CLI tool in a virtual environment managed by Poetry.

## Usage

There are three steps: frame extraction, image classification, and showing
the images containing the class of interest.

### 1 Frame extraction

To extract frames from `myvideo.mp4` in your working directory:

```bash
poetry run object-detect extract-frames myvideo.mp4 frames/
```

This will extract frames into `frames/`. The frames are saved as
JPEGs by default.

### 2 Image classification

To classify all the images in `frames/`:

```bash
poetry run object-detect detect frames/
```

The prediction results are written to `predictions.jsonl`.
Each line in this [JSONL](https://jsonlines.org/) contains
the prediction result for a single frame.

### 3 Showing images containing a class

Finally, to print the paths of images containing a certain class
(e.g., "cat"):

```bash
poetry run object-detect show predictions.jsonl cat
```

### Options

For more CLI options:

```bash
poetry run object-detect --help
poetry run object-detect extract-frames --help
poetry run object-detect detect --help
poetry run object-detect show --help
```

Tip: to run the extraction in parallel for multiple videos, use `xargs`
with `-P`. For example:

```bash
find "/path/to/videos" -name "*.mp4" | xargs -P $(nproc) -I % poetry run object-detect extract-frames % %.frames
```

## Development

Follow the installation instructions above. To run the tests:

```bash
./test.sh
```

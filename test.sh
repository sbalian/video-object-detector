#!/bin/bash

set -x

poetry run mypy --install-types --non-interactive video_object_detector
poetry run black --check video_object_detector
poetry run isort --check video_object_detector
poetry run flake8 video_object_detector
poetry run pytest

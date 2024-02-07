#!/bin/bash

set -x

poetry run mypy --install-types --non-interactive object_detector
poetry run black --check object_detector
poetry run isort --check object_detector
poetry run flake8 object_detector
poetry run pytest

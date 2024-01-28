#!/bin/bash

poetry run mypy --install-types --non-interactive bicycle_detector
poetry run black --check bicycle_detector
poetry run isort --check bicycle_detector
poetry run flake8 bicycle_detector

#!/bin/bash -eu

BASE_DIR=$(realpath $(dirname $(dirname $0)))

PYTHONPATH="${BASE_DIR}" python ${BASE_DIR}/src/evaluate.py ${BASE_DIR}/models/dwk_pretrained.pt --data-folder ${BASE_DIR}/dwk_data

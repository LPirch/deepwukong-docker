#!/bin/bash -eu

BASE_DIR=$(realpath $(dirname $(dirname $0)))

# train surrogate model
PYTHONPATH="${BASE_DIR}" python ${BASE_DIR}/src/run.py -c configs/surrogate.yaml
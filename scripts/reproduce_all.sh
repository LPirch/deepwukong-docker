#!/bin/bash -eu

BASE_DIR=$(realpath $(dirname $(dirname $0)))

# evaluate models on preprocessed data
${BASE_DIR}/scripts/eval_preprocessed.sh

# re-extract data using joern
# TODO
${BASE_DIR}/scripts/eval_raw.sh || echo "skipping raw evaluation (not implemented yet)"
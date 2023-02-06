#!/bin/bash -eu

BASE_DIR=$(realpath $(dirname $(dirname $0)))

${BASE_DIR}/scripts/eval_pretrained.sh
${BASE_DIR}/scripts/repro_preprocessed.sh
${BASE_DIR}/scripts/repro_raw.sh

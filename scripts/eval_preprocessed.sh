#!/bin/bash -eu

BASE_DIR=$(realpath $(dirname $(dirname $0)))

# the published evaluation script on pretrained model with preprocessed data
PYTHONPATH="${BASE_DIR}" python ${BASE_DIR}/src/evaluate.py ${BASE_DIR}/models/dwk_pretrained.pt --data-folder ${BASE_DIR}/data

# re-train the original model
PYTHONPATH="${BASE_DIR}" python ${BASE_DIR}/src/run.py -c configs/dwk.yaml

# train surrogate model
PYTHONPATH="${BASE_DIR}" python ${BASE_DIR}/src/run.py -c configs/surrogate.yaml

# train simple surrogate model (MLP)
PYTHONPATH="${BASE_DIR}" python ${BASE_DIR}/src/run.py -c configs/simple_mlp.yaml

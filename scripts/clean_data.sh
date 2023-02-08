#!/bin/bash -eu

BASE_DIR=$(realpath $(dirname $(dirname $0)))
DATA_DIR=${BASE_DIR}/data

# delete generated files
for target in CWE119/XFG CWE119/csv CWE119/done.txt; do
    if [ -d $DATA_DIR/$target ] || [ -f $DATA_DIR/$target ]; then
        rm -r $DATA_DIR/$target
    fi
done

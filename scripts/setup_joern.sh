#!/bin/bash -eu

BASE_DIR=$(realpath $(dirname $(dirname $0)))

if ! command -v git gradle &> /dev/null; then
    echo "ERROR: missing $cmd command"
    exit 1
fi

JOERN_URL="https://github.com/ives-nx/dwk_preprocess.git"


# clone deprecated version of joern
tmp_dir="${BASE_DIR}/.dwk-preprocess"
git clone $JOERN_URL ${tmp_dir}
mv ${tmp_dir}/joern_slicer/joern ${BASE_DIR}
rm -rf ${tmp_dir}
pushd ${BASE_DIR}/joern
chmod u+x build.sh
./build.sh
chmod u+x joern-parse
popd

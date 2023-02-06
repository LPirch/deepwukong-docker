#!/bin/bash -eu

BASE_DIR=$(realpath $(dirname $(dirname $0)))

# check for dependencies
for cmd in 7z wget realpath; do
    if ! command -v $cmd &> /dev/null; then
        echo "ERROR: missing $cmd command"
        exit 1
    fi
done

DATA_URL="https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EalnVAYC8zZDgwhPmGJ034cBYNZ8zB7-mNSNm-a7oYXkcw\?download\=1"
MODEL_URL="https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EesTvivx1UlEo9THYRSCYkMBMsZqKXgNVYx9wTToYnDwxg\?download\=1"

wget -O ${BASE_DIR}/data/data.7z ${DATA_URL}
mkdir -p ${BASE_DIR}/models
wget -O ${BASE_DIR}/models/dwk_pretrained.pt ${MODEL_URL}

# extract data
pushd ${BASE_DIR}/data
7z x data.7z && rm data.7z
popd

# copy data to allow parallel reproduction 
# from preprocessed and self-processed data/models
cp -r ${BASE_DIR}/data {BASE_DIR}/dwk_data
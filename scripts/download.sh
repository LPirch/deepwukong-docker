#!/bin/bash -eu

BASE_DIR=$(realpath $(dirname $(dirname $0)))

# check for dependencies
for cmd in 7z wget realpath git; do
    if ! command -v $cmd &> /dev/null; then
        echo "ERROR: missing $cmd command"
        exit 1
    fi
done

DATA_URL=https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EalnVAYC8zZDgwhPmGJ034cBYNZ8zB7-mNSNm-a7oYXkcw\?download\=1
MODEL_URL=https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EesTvivx1UlEo9THYRSCYkMBMsZqKXgNVYx9wTToYnDwxg\?download\=1

mkdir -p ${BASE_DIR}/data
mkdir -p ${BASE_DIR}/models

wget -O ${BASE_DIR}/data.7z ${DATA_URL}
wget -O ${BASE_DIR}/models/dwk_pretrained.pt ${MODEL_URL}


# extract data
pushd ${BASE_DIR}/data
7z x ../data.7z
popd

rm ${BASE_DIR}/data.7z

# deepwukong-docker

Reproducing results of the following paper using docker:

> (TOSEM'21) DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network

## Adjustments

This is an overview about the applied adjustments:

- needed to update the following dependencies (to work with my cuda version):
  -  torch from `1.9.0` to `1.13.0`
  -  torch-scatter from `2.0.7` to `2.1.0`
  -  torch-sparse from `0.6.10` to `0.6.16`
-  added: write test results to JSON file
-  disabled: cudnn support (didn't work on my hardware)


## Reproduction (docker) Setup and Usage

```shell
docker build . -t dwk
docker run --rm --gpus '"device=0,3"' -v $(realpath data):/root/dwk/data -v $(realpath dwk_data):/root/dwk/dwk_data dwk ./scripts/reproduce_all.sh  # change device IDs as needed

```

## Setup

- Environment

    ```shell
    bash env.sh
    ```

- Preprocessed Data

    Download from [data](https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EalnVAYC8zZDgwhPmGJ034cBYNZ8zB7-mNSNm-a7oYXkcw?e=eRUc50), and unzip the data under `<project root>/data` folder.

---

## One-Step Evaluation

- From Pretrained model
  
  - Download from [pretrained model](https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EesTvivx1UlEo9THYRSCYkMBMsZqKXgNVYx9wTToYnDwxg?e=Z4nz23).
  - `PYTHONPATH="." python src/evaluate.py <path to the pretrained model>`

- Training and Testing

  ```shell
  bash run.sh
  ```

---

**Run from scratch:**

## Data preparation

### Use joern to Generate PDG

**We use the old version of [joern](https://github.com/ives-nx/dwk_preprocess/tree/main/joern_slicer/joern) to generate PDG**

```shell
PYTHONPATH="." python src/joern/joern-parse.py -c <config file>
```

### Generate raw XFG

```shell
PYTHONPATH="." python src/data_generator.py -c <config file>
```

### Symbolize and Split Dataset

```shell
PYTHONPATH="." python src/preprocess/dataset_generator.py -c <config file>
```

### Word Embedding Pretraining

```shell
PYTHONPATH="." python src/preprocess/word_embedding.py -c <config file>
```

## Evaluation

```shell
PYTHONPATH="." python src/run.py -c <config file>
```


## Citation

Please kindly cite our paper if it benefits:

```bib
@article{xiao2021deepwukong,
author = {Cheng, Xiao and Wang, Haoyu and Hua, Jiayi and Xu, Guoai and Sui, Yulei},
title = {DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network},
year = {2021},
publisher = {ACM},
volume = {30},
number = {3},
url = {https://doi.org/10.1145/3436877},
doi = {10.1145/3436877},
journal = {ACM Trans. Softw. Eng. Methodol.},
articleno = {38},
numpages = {33}
}
```

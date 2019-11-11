This repository contains the implementation of `Fine-grained Privacy Detection with Graph-regularized Hierarchical Attentive Representation Learning` in tensorflow.

# Requirements:
* Python 3.5.2
* Tensorflow 1.9.0
* Numpy 1.16.4
* Nltk

# Dataset:
We conducted our experiments on the public real-world dataset introduced in [1], which consists of 11,368 tweets annotated with 32 personal aspects.

# Usage:
* Download the dataset and unzip it into `data` folder. 
* Train and evaluate the model: 
```Python
python train.py
```

# Reference:
[1] Xuemeng Song et al. A personal privacy preserving framework: I let you know who can see what. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 295–304, 2018.

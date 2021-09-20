# JUNO 能量重建实验报告

**Team: PMTMender**

**Members: 刘明昊 沈若寒  赵海萌**

**小组主要分工：**

- 沈若寒：
- 刘明昊：
- 赵海萌：

## 摘要

本项目以JUNO中微子探测装置为背景，试图通过物理分析、信号处理与机器学习等技术，从PMT波形中重建出中微子事件的能量。通过数据预处理、特征工程和多级LightGBM的统计学习，我们的算法能在很短的时间内，以很高的精度重构出事件的能量，同时具备很好的可解释性。高效率、高精度、可解释的能量重建手段有助于我们理解中微子质量顺序的难题。

## 目录

[TOC]

## 整体思路



## 0. 文件结构与执行方式

### 0.1. 文件结构

本项目的文件结构如下：

```
|-- project-1-junosap-pmtmender
		|-- README.md
    |-- requirements.txt
    |-- Makefile
    |-- utils.py
    |-- waveform.py
    |-- model.ipynb
    |-- train.ipynb
    |-- final.ipynb
```

其中`README.md` 为本实验报告，`requirements.txt` 罗列了本项目的依赖包版本，`Makefile` 定义了处理流水线， `waveform.py` 进行数据预处理，`train.ipynb` 训练模型，`final.ipynb` 生成预测答案。

### 0.2. 执行方式

在执行前请确保依赖包都已安装并符合版本要求，执行

```shell
pip install -r requirements.txt
```

以安装依赖包。

在项目目录下用 `shell` 执行代码

```shell
make
```

可以完整地执行整个项目的流程，下载训练集、预处理数据、训练模型、生成预测答案。

执行代码

```shell
make data
make train
make model
make ans
```

可分别下载数据、预处理数据、训练模型、生成预测答案。

执行代码

```shell
make clean
```

可以清理下载的数据集、预处理数据和训练模型，但不会清理预测答案。

若要单独执行预处理数据，可以执行

```shell
python3 waveform.py
```

若要单独预处理数据和训练模型，可以执行`train.ipynb`和`final.ipynb`。

## 1. 顶点模拟与光子生成

### 1.1. 思路



### 1.2. 主要实现方式

#### 1.2.1. 顶点坐标生成

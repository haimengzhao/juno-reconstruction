# JUNO 能量重建实验报告

**Team: PMTMender**

**Members: 刘明昊 沈若寒  赵海萌**

**小组主要分工：**

- 沈若寒：
- 刘明昊：
- 赵海萌：Ghost Hunter Legacy，撰写报告

**注意！** 本报告仅是一个简短的说明文档和操作手册，详细实现思路与细节请见`model.ipynb, train.ipynb, final.ipynb` 中的Markdown说明。

## 摘要

本项目以JUNO中微子探测装置为背景，试图通过物理分析、信号处理与机器学习等技术，从PMT波形中重建出中微子事件的能量。通过数据预处理、特征工程和多级LightGBM的统计学习，我们的算法能在很短的时间内，以很高的精度重构出事件的能量，同时具备很好的可解释性。高效率、高精度、可解释的能量重建手段有助于我们理解中微子质量顺序的难题。

## 目录

[TOC]

## 整体思路

我们面临的问题是要从波形中重构出事件能量。考虑到我们拥有大量的训练数据，这是一个典型的机器学习回归问题。其中的困难主要在于**特征工程**：即如何从波形中提取出有效的特征，用于训练预测能量的模型。

为了保证较高的效率、可复现性与可解释性，避免过大的计算资源需求，我们没有采取以BERT为首的一系列深度神经网络，而是采用了Kaggle等数据科学竞赛中常用的传统机器学习算法LightGBM，其具有**效率高、迭代快、效果好**的特点，方便我们进行调整并测试不同的特征工程方法与超参数。

那么余下的问题便是特征工程，如何从巨量数据的波形中提取出有效的特征。考虑到击中每个波形的PE个数是一个重要的中间量，我们首先将整个波形-能量的任务拆分成两步：波形-每个波形的PE个数和PE个数-能量。

对于这两个步骤，我们分别提取特征进行模型训练（具体特征工程见`model.ipynb`），最后合成成为多级LightGBM分步进行预测。

## 文件结构

本项目的文件结构如下：

```
|-- project-1-junosap-pmtmender
		|-- README.md
    |-- requirements.txt
    |-- Makefile
    |-- utils.py
    |-- waveform.py
    |-- train.ipynb
    |-- final.ipynb
    |-- model.ipynb
```

其中`README.md` 为本实验报告，`requirements.txt` 罗列了本项目的依赖包版本，`Makefile` 定义了处理流水线， `waveform.py` 进行数据预处理，`train.ipynb` 训练模型，`final.ipynb` 生成预测答案。

**注意**：`model.ipynb` 是一个简要的示例代码，阅读它可以使读者更好地了解我们的算法。

## 执行方式

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

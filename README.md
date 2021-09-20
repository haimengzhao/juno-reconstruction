# JUNO 能量重建实验报告

**Team: PMTMender**

**Members: 刘明昊 沈若寒  赵海萌**

**小组主要分工：**

- 沈若寒：
- 刘明昊：
- 赵海萌：

## 摘要



## 目录

[TOC]

## 整体思路



## 0. 文件结构与执行方式

### 0.1. 文件结构

本项目的文件结构如下：

```
|-- project-1-junosap-pmtmender
    |-- requirements.txt
```

其中`requirements.txt` 罗列了本项目的依赖包版本

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

可以完整地执行整个项目的流程，生成模拟数据文件 `data.h5` 并根据该数据绘制图像 `figures.pdf` 。

执行代码

```shell
make data.h5
make figures.pdf
make probe.pdf
```

可分别生成模拟数据 `data.h5` 、绘图 `figures.pdf` 以及Sim-driven的Probe函数热力图 `probe.pdf` 。

执行代码

```shell
make clean
```

可以清理生成的 `data.h5` 、 `figures.pdf` 和 `probe.pdf` 文件。

若要单独测试 `simulate.py` 和 `draw.py` ，可以执行

```shell
python3 simulate.py -n <num_of_events> -g geo.h5 -o <output_file> [-p <num_of_pmts>]
python3 draw.py <data_file> -g geo.h5 -o <output_file>
```

例如：

```shell
python3 simulate.py -n 4000 -g geo.h5 -o data.h5
python3 draw.py data.h5 -g geo.h5 -o figures.pdf
```

## 1. 顶点模拟与光子生成

### 1.1. 思路



### 1.2. 主要实现方式

#### 1.2.1. 顶点坐标生成

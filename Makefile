.PHONY: all clean data train model ans

indices:=$(shell seq 2 19)
dataFile:=$(indices:%=./data/final-%.h5) ./data/final.h5
trainFile:=$(indices:%=./train/final_%_wf.h5) ./train/final_wf.h5
modelFile:=./model/modelPCalc.txt

# 默认的目标是ans
all: ans

# 删除train, model文件夹
clean:
	rm -rf train model

# 下载数据
data: ${dataFile}

${dataFile} &:
	./fetch.sh

# 预处理数据
train: ${trainFile}

${trainFile} &: ${dataFile}
	mkdir -p train
	python3 waveform.py

# 训练决策树
model: ${modelFile}

${modelFile}: ${trainFile}
	mkdir -p model
	jupyter nbconvert --to=python train.ipynb
	ipython3 train.py
	rm -f train.py

# 得到答案
ans: ${modelFile}
	mkdir -p ans
	touch ans/log
	./prompt.sh
	jupyter nbconvert --to=python final.ipynb
	ipython3 final.py
	rm -f final.py

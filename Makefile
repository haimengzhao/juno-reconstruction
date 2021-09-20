.PHONY: all clean

all: figures.pdf

clean:
	rm -rf data train model

data:
	wget -P ./data/ http://hep.tsinghua.edu.cn/~orv/dc/bdeph2021/geo.h5;
	for num in $$(seq 2 19); do echo $${num}; wget -P ./data/ http://hep.tsinghua.edu.cn/~orv/dc/bdeph2021/final-$${num}.h5; done
	wget -P ./data/ http://hep.tsinghua.edu.cn/~orv/dc/bdeph2021/final.h5;

train: data
	python3 waveform.py

model: train
	jupyter nbconvert --to=python train.ipynb
	python3 train.py
	rm -f train.py

ans: model
	jupyter nbconvert --to=python train.ipynb
	python3 final.py
	rm -f final.py

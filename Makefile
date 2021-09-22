.PHONY: all clean data train model ans

indices:=$(shell seq 2 19)
dataFile:=$(indices:%=./data/final-%.h5) ./data/final.h5
trainFile:=$(indices:%=./train/final_%_wf.h5) ./train/final_wf.h5
modelFile:=./model/modelPCalc.txt
ansFile:=./ans/ans17.h5

all: ans

clean:
	rm -rf ${trainFile} ${modelFile} ${ansFile}

data: ${dataFile}

${dataFile} &:
	./fetch.sh

train: ${trainFile}

${trainFile} &: ${dataFile}
	python3 waveform.py

model: ${modelFile}

${modelFile}: ${trainFile}
	jupyter nbconvert --to=python train.ipynb
	ipython3 train.py
	rm -f train.py

ans: ${ansFile}

${ansFile}: ${modelFile}
	jupyter nbconvert --to=python final.ipynb
	ipython3 final.py
	rm -f final.py

#!/bin/bash

# 本脚本用于下载数据

fileArray=(geo.h5 final.h5)

for num in $(seq 2 19)
do
	fileArray[$num]=final-${num}.h5
done

for file in ${fileArray[@]}
do
	if ! test -f ./data/${file}
	then
		wget -P ./data/ http://hep.tsinghua.edu.cn/~orv/dc/bdeph2021/${file}
	fi
done

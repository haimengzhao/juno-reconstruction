#!/bin/bash

# 本脚本用于下载数据

fileArray=(geo.h5 final.h5)

# 更新fileArray，加入final-*.h5
for num in $(seq 2 19)
do
	fileArray[$num]=final-${num}.h5
done

# 下载数据
for file in ${fileArray[@]}
do
	# 当目录中没有这个文件才下载
	if ! test -f ./data/${file}
	then
		wget -P ./data/ http://hep.tsinghua.edu.cn/~orv/dc/bdeph2021/${file}
	fi
done

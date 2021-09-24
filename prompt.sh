#!/bin/bash

# 此脚本用于设置答案的文件名序号和log
# 答案的文件名格式为ans${num}.h5，其中num是正整数
# log将被写入到./ans/log中

echo "请输入答案的文件名序号:"
read num

# 判断num是否合法，以及文件是否已存在
while echo $num | grep -q [^0-9] || [[ $num -le 0 ]] || [[ -e ./ans/ans${num}.h5 ]]
do
	if [[ -e ./ans/ans${num}.h5 ]]
	then
		echo "文件已存在，请重新输入:"
	else
		echo "序号必须是大于0的整数，请重新输入:"
	fi
	read num
done

echo "请输入本次运行的log:"
read log

# 修改final.ipynb中的输出路径
sed -E "s/ans[0-9]+\.h5/ans${num}.h5/g" -i ./final.ipynb

# 将log输出到./ans/log
echo "ans${num}.h5: $log" >> ./ans/log

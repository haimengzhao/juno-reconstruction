#!/bin/bash

echo "请输入答案的文件名序号:"
read num

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

sed -E "s/ans[0-9]+\.h5/ans${num}.h5/g" -i ./final.ipynb
echo "ans${num}.h5: $log"

import multiprocessing
import os
import warnings
import numpy as np
import h5py as h5
from tqdm import tqdm
import utils

def preprocessWF(trainWF):
    '''
    preprocessWF: 预处理波形
    
    输入: trainWF: (n, 1000) ndarray

    输出：波形积分值(n,) ndarray
          超过阈值的点数(n,) ndarray
          手作算法得到的每个波形PE数(n,) ndarray
          手作算法得到的每个波形PETime(n,) ndarray
    '''
    print("正在去除波形噪音...")
    denoisedTrainWF = np.empty((trainWF.shape[0], 1000), dtype='<i2')
    step = 2000000
    for i in tqdm(range(trainWF.shape[0] // step + 1)):
        denoisedTrainWF[step*i:step*(i+1)] = np.where(
            trainWF['Waveform'][step*i:step*(i+1), :] < 918,
            918-trainWF['Waveform'][step*i:step*(i+1), :],
            0
        )
    print("正在做波形积分...")
    intTrainWF = np.sum(denoisedTrainWF, axis=1)
    print("正在计算超出阈值的点数...")
    pointsPerTrainWF = np.sum(denoisedTrainWF > 0, axis=1)
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with multiprocessing.Pool(8) as p:
            res = np.array(
                list(
                    tqdm(
                        p.imap(
                            utils.getPePerWF,
                            denoisedTrainWF
                        ),
                        total=denoisedTrainWF.shape[0]
                    )
                )
            )
    '''
    res = np.empty((2, trainWF.shape[0]))
    step = 100000
    for i in tqdm(range(trainWF.shape[0] // step + 1)):
        res[:, step*i:step*(i+1)] = utils.getPePerWF(denoisedTrainWF[step*i:step*(i+1)])
    pePerTrainWFCalc = res[0]
    meanPeTimePerTrainWF = res[1]
    return intTrainWF, pointsPerTrainWF, pePerTrainWFCalc, meanPeTimePerTrainWF

def getPePerTrainWF(trainPET, trainWF):
    '''
    getPePerTrainWF: 从PETruth与Waveform中得到真实的每个波形对应的PE数

    输入: trainPET，文件中的PETruth表
          trainWF, 文件中的Waveform表

    输出: 真实的每个波形对应的PE数(n,) ndarray
    '''
    print("正在得到训练集的目标pePerWF...")
    numPET, peIndices = utils.getNum(trainPET)
    numPEW, wfIndices = utils.getNum(trainWF)
    pePerTrainWF = np.array([])
    splitWFChannels = np.split(trainWF['ChannelID'], wfIndices[1:-1])
    for index, arr in enumerate(tqdm(np.split(trainPET['ChannelID'], peIndices[1:-1]))):
        channels, counts = np.unique(arr, return_counts=True)
        zeroPeChannelCount = numPEW[index].astype(int) - channels.shape[0]
        while zeroPeChannelCount:
            alignedChannels = np.append(channels, np.zeros(zeroPeChannelCount)-1)
            indexToInsert = np.asarray(alignedChannels != splitWFChannels[index]).nonzero()[0][0]
            channels = np.insert(channels, indexToInsert, splitWFChannels[index][indexToInsert])
            counts = np.insert(counts, indexToInsert, 0)
            zeroPeChannelCount -= 1

        pePerTrainWF = np.append(pePerTrainWF, counts)

    return pePerTrainWF.flatten().astype(int)

if __name__ == '__main__':
    # 数据集所在位置
    trainpathRoot = './data/final-'
    problemDType = [
        ('intWF', '<i4'),
        ('pointsPerWF', '<i2'),
        ('pePerWFCalc', '<i2'),
        ('meanPeTimePerWF', '<f8')
    ]
    trainDType = [
        ('intWF', '<i4'),
        ('pointsPerWF', '<i2'),
        ('pePerWFCalc', '<i2'),
        ('meanPeTimePerWF', '<f8'),
        ('pePerWF', '<i2')
    ]


    # 先处理题目
    if True:# not os.path.exists(f"./train/final_wf.h5"):
        print("下面处理题目...")
        trainWF = utils.loadData(f"./data/final.h5", 'test')
        intTrainWF, pointsPerTrainWF, pePerTrainWFCalc, meanPeTimePerTrainWF = preprocessWF(trainWF)
        _, wfIndices = utils.getNum(trainWF)

        # 存储文件
        data = np.zeros(
            intTrainWF.shape[0],
            dtype=problemDType
        )
        data['intWF'] = intTrainWF
        data['pointsPerWF'] = pointsPerTrainWF
        data['pePerWFCalc'] = pePerTrainWFCalc
        data['meanPeTimePerWF'] = meanPeTimePerTrainWF
        
        with h5.File('./train/final_wf.h5', 'w') as opt:
            opt['Waveform'] = data
            opt['WfIndices'] = wfIndices


    # 循环处理训练集
    for i in range(2, 20):
        print(f"下面处理final-{i}.h5...")
        if False:#os.path.exists(f"./train/final_{i}_wf.h5"):
            print(f"final-{i}.h5已经处理过了，继续...")
            continue
        trainPET, trainWF, trainPT = utils.loadData(f"{trainpathRoot}{i}.h5", 'PT')
        intTrainWF, pointsPerTrainWF, pePerTrainWFCalc, meanPeTimePerTrainWF = preprocessWF(trainWF)
        pePerTrainWF = getPePerTrainWF(trainPET, trainWF)
        _, wfIndices = utils.getNum(trainWF)

        # 存储文件
        data = np.zeros(
            intTrainWF.shape[0],
            dtype=trainDType
        )
        data['intWF'] = intTrainWF
        data['pointsPerWF'] = pointsPerTrainWF
        data['pePerWFCalc'] = pePerTrainWFCalc
        data['meanPeTimePerWF'] = meanPeTimePerTrainWF
        data['pePerWF'] = pePerTrainWF
        
        with h5.File(f"./train/final_{i}_wf.h5", 'w') as opt:
            opt['Waveform'] = data
            opt['WfIndices'] = wfIndices


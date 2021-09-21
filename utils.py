import numpy as np
import h5py

def lossfunc_eval(y, data):
    t = data.get_label()
    loss = (y - t) ** 2 / t
    return "custom", np.mean(loss), False

def lossfunc_train(y, data):
    t = data.get_label()
    grad = 2 * (y - t) / t
    hess = 2 * np.ones_like(y) / t
    return grad, hess

def loadData(datapath, datatype='geo'):
    with h5py.File(datapath, "r") as in_file: 
        # 如果不是pro，则读取的是数据集h5文件，打印文件结构
        if datatype != 'pro':
            print("Structure of data:")
            for key in in_file.keys():
                print(in_file[key], key, in_file[key].name)
        
        # 读取geo.h5
        if datatype == 'geo':
            return in_file['Geometry'][...]
        
        # 读取final-${num}.h5
        if datatype == 'PT':
            PT = in_file['ParticleTruth'][...]
            PET = in_file['PETruth'][...]
            WF = in_file['Waveform'][...]
            return PET, WF, PT
        
        # 读取训练需要的数据
        if datatype=='pro':
            X = np.array(in_file['X'])
            Y = np.array(in_file['Y'])
            return X, Y
        
        # 读取ParticleTruth
        if datatype == 'p':
            PT = in_file['ParticleTruth'][...]
            return PT
        
        # PET = np.array(np.array(file['PETruth']).tolist())
        # 如果是另外的datatype，则返回Waveform
        WF = in_file['Waveform'][...]
        return WF

def getNum(dataset):
    '''
    getNum: 返回每个EventID对应的PE数或波形数。
    
    输入：Dataset，structured array, 形状为(n,)
         传入的既可以是Waveform表，也可以是PETruth表，但一定要有EventID列，且按照顺序排列。
    输出：num, ndarray, 形状为(m,)，下标为i对应的数为EventID=i时对应的PE数或波形数
         indices, ndarray, 形状为(m,), unique后返回的index
    '''
    eventIDs, indices = np.unique(dataset['EventID'], return_index=True)
    indices = np.append(indices, dataset.shape[0])
    num = np.diff(indices)
    return num, indices

def getPePerWF(waveform):
    '''
    
    '''
    def getCancel(maxIndex, maxValue):
        '''
        getCancel: 返回一个中心在(maxIndex, maxValue)的绝对值函数，小于0的地方变为0，宽度为16
        '''
        step = maxValue / 8
        absArray = maxValue - np.abs(np.arange(1000) - maxIndex.reshape(-1, 1)) * step
        return np.where(absArray > 0, absArray, 0)

    cancelledWF = waveform
    integrate = np.sum(cancelledWF, axis=1)
    points_more_than_threshold = np.sum(cancelledWF > 0, axis=1)
    noise_uncancelled_region = np.zeros(waveform.shape).astype(bool)
    peCount = np.zeros(waveform.shape[0], dtype=int)
    peFilteredCount = np.zeros(waveform.shape[0], dtype=int)
    peTimeSum = np.zeros(waveform.shape[0])
    label = points_more_than_threshold >= 4

    while np.any(label):
        argmax = np.argmax(cancelledWF[label, :], axis=1)
        toCancel = getCancel(argmax, 18)
        cancelledWF[label, :] -= np.round(toCancel).astype(int)
        noise_uncancelled_region[label, :] = np.logical_or(noise_uncancelled_region[label, :], toCancel)
        judge_noise = cancelledWF[noise_uncancelled_region]
        # if np.sum(judge_noise) < 2*noise_uncancelled_region.shape[0]:
        judge_noise = np.where(judge_noise < 8, 0, judge_noise)
        cancelledWF[noise_uncancelled_region] = judge_noise
        
        integrate = np.sum(cancelledWF, axis=1)
        points_more_than_threshold = np.sum(cancelledWF > 0, axis=1)
        
        peCount[label] += 1
        peFilteredCount[label] += np.all([argmax <= 600, argmax >= 150], axis=0)
        peTimeSum[label] += argmax*np.all([argmax <= 600, argmax >= 150], axis=0)

        label = np.logical_and(label, integrate >= 150 - 8*np.maximum(points_more_than_threshold, 16-points_more_than_threshold))

    return peCount, peTimeSum / peFilteredCount

def saveData(X, Y, path):
    h5 = h5py.File(path,'w')
    dataX = h5.create_dataset(name='X', data=X)
    dataY = h5.create_dataset(name='Y', data=Y)
    h5.close()


def saveans(ans, path):
    h5 = h5py.File(path,'w')
    A = h5.create_dataset(name='Answer', shape=(4000, ), dtype=np.dtype([('EventID', 'i'), ('p', 'f')]))
    A["EventID"] = np.arange(4000)
    A["p"] = ans
    h5.close()


# wf (n, 1000) int
# label (n,) bool 
# allIndex = np.arange(2000000)
# PEnum = (n,)
# allPETime = (n,)

# while (label == True).any:
#     needManage = allIndex[label]   #需要减去PE的波形的index
#     wfAfter, times = manage(wf[needManage])   #处理需要减去PE的波形,返回处理完的波形和判断出的时间
#     wf[needManage] = wfAfter  #把处理完的波形放回wf
#     PEnum[needManage] = PEnum[needManage] + 1  #增加探测到的光子数
#     allPETime[needManage] += times

#     wellManaged = ifwellManaged(wfAfter)   #判断处理完的波形是否符合了要求（要求：不需要再处理），返回一个mask，needManage中符合条件的位置为True
#     doneIndex = needManage[wellManaged]  #返回那些不需要再搞的波形在wf中的index
#     label[doneIndex] = np.bitwise_not(label[doneIndex]) #将那些搞定了的wf的label设置为False

# return PEnum, allPETime/PEnum

import numpy as np
import h5py
from multiprocessing import Pool
from tqdm import tqdm

def lossfunc_eval(y, data):
    '''
    lossfunc_eval: 用平台的评分标准来评估决策树
    '''
    t = data.get_label()
    loss = (y - t) ** 2 / t
    return "custom", np.mean(loss), False

def lossfunc_train(y, data):
    '''
    lossfunc_train: 用平台的评分标准算梯度与hessian矩阵
    '''
    t = data.get_label()
    grad = 2 * (y - t) / t
    hess = 2 * np.ones_like(y) / t
    return grad, hess

def read(ids):
    '''
    read: 强行变成多进程用的内部函数
    '''
    return in_file['Waveform'][ids]

def loadData(datapath, datatype):
    '''
    loadData: 读取h5文件
    输入：datapath，字符串，表示文件路径
         datatype，字符串，有以下选择：
                   geo: 读取几何文件
                   PT:  读取训练集中的所有表，以PETruth, Waveform, ParticleTruth的顺序返回
                   p:   读取训练集中的ParticleTruth表
                   其他: 读取其中的Waveform表
    输出：若干个structured array
    '''
    global in_file
    with h5py.File(datapath, "r") as in_file:

        # 读取geo.h5
        if datatype == 'geo':
            return in_file['Geometry'][...]
        
        # 读取final-${num}.h5
        if datatype == 'PT':
            PT = in_file['ParticleTruth'][...]
            PET = in_file['PETruth'][...]

            # 多进程读取Waveform
            WF = np.empty(in_file['Waveform'].shape, dtype=in_file['Waveform'].dtype)
            chunkNum = 16
            indices = np.array_split(np.arange(in_file['Waveform'].shape[0]), chunkNum)
            with Pool(4) as pool:
                print("正在读取data")
                WF = np.concatenate(list(tqdm(pool.imap(read, indices), total=chunkNum)))
            return PET, WF, PT

        # 读取ParticleTruth
        if datatype == 'p':
            PT = in_file['ParticleTruth'][...]
            return PT
        
        # 如果datatype是另外的，则返回Waveform表
        WF = np.empty(in_file['Waveform'].shape, dtype=in_file['Waveform'].dtype)
        chunkNum = 16
        indices = np.array_split(np.arange(in_file['Waveform'].shape[0]), chunkNum)
        with Pool(4) as pool:
            print("正在读取data")
            WF = np.concatenate(list(tqdm(pool.imap(read, indices), total=chunkNum)))
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


def getCancel(maxIndex, maxValue):
    '''
    getCancel: 返回一个峰值在(maxIndex, maxValue)的三角脉冲波，宽度为16
    '''
    step = maxValue / 8
    absArray = maxValue - np.abs(np.arange(1000) - maxIndex.reshape(-1, 1)) * step
    return np.where(absArray > 0, absArray, 0)

def getPePerWF(waveform):
    '''
    getPePerWF: 手作算法，处理波形，得到除去部分暗噪声的PE总数和PETime平均值（以argmax来替代真正的PETime）
    
    输入: waveform, (n, 1000)ndarray, 表示n个除噪声、反向之后的waveform
    输出: result, (2, n)ndarray, result[0]表示PE总数，result[1]表示PETime平均值
    '''
    
    # 预处理
    cancelledWF = waveform
    integrate = np.sum(cancelledWF, axis=1)
    points = np.sum(cancelledWF > 0, axis=1)
    peCount = np.zeros(waveform.shape[0], dtype=int)
    peFilteredCount = np.zeros(waveform.shape[0], dtype=int)
    peTimeSum = np.zeros(waveform.shape[0])
    
    label = np.where(points >= 4)[0] # label是要处理的波形的下标
    cancelledWF = np.take(cancelledWF, label, axis=0) # 取出要处理的波形
    noiseUncancelledRegion = np.zeros(cancelledWF.shape, dtype=bool) # 存储之前减过不为0的取消函数的区域

    # 预先算好1000个cancel函数
    cancels = np.round(getCancel(np.arange(1000), 18)).astype(int)

    while label.shape[0]:
        # 先找出argmax，再减去取消函数
        argmax = np.argmax(cancelledWF, axis=1)
        toCancel = np.take(cancels, argmax, axis=0)
        np.subtract(cancelledWF, toCancel, out=cancelledWF)
        
        # 判断减去后的波形是否是由噪声导致的
        noiseUncancelledRegion = np.logical_or(noiseUncancelledRegion, toCancel>0)
        judgeNoise = cancelledWF[noiseUncancelledRegion]
        judgeNoise = np.where(judgeNoise < 8, 0, judgeNoise)
        cancelledWF[noiseUncancelledRegion] = judgeNoise
        
        # 重新计算积分与超过0的点数量
        integrate = np.sum(cancelledWF, axis=1)
        points = np.sum(cancelledWF > 0, axis=1)
        
        # 去除暗噪声，刷新PE总数与PETime之和
        peFilteredCount[label] += np.all([argmax <= 600, argmax >= 150], axis=0)
        peTimeSum[label] += argmax*np.all([argmax <= 600, argmax >= 150], axis=0)

        # 判断新的要处理的波形
        newLabelIndex = integrate >= 150 - 8*np.maximum(points, 16-points)
        label = np.compress(newLabelIndex, label, axis=0)
        cancelledWF = np.compress(newLabelIndex, cancelledWF, axis=0)
        
        # 重置noiseUncancelledRegion
        noiseUncancelledRegion = np.compress(newLabelIndex, noiseUncancelledRegion, axis=0)

    return np.stack((peFilteredCount, peTimeSum / peFilteredCount))


def saveans(ans, path):
    '''
    saveans: 按照标准格式保存答案
    
    输入: ans, ndarray, [4000,]，表示计算得到的p值
         path, 字符串, 表示要存储的文件位置
    '''
    with h5py.File(path,'w') as h5:
        A = h5.create_dataset(name='Answer', shape=(4000, ), dtype=np.dtype([('EventID', '<i4'), ('p', '<f8')]))
        A["EventID"] = np.arange(4000)
        A["p"] = ans
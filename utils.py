import numpy as np
import h5py
import pywt

# 小波滤噪
def wavelet_denoising(data):
    # 小波函数取db4
    db4 = pywt.Wavelet('db4')
    # 分解
    coeffs = pywt.wavedec(data, db4)
    # 高频系数置零
    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    coeffs[len(coeffs)-3] *= 0
    # 重构
    meta = pywt.waverec(coeffs, db4)
    return meta

def lossfunc_eval(y, data):
    t = data.get_label()
    loss = (y - t) ** 2 / t
    return "custom", np.mean(loss), False

def lossfunc_train(y, data):
    t = data.get_label()
    grad = 2 * (y - t) / t
    hess = 2 * np.ones_like(y) / t
    return grad, hess

def tkeo(arr):
    tkeo = np.copy(arr)
    # Teager–Kaiser Energy operator
    tkeo[1:-1] = arr[1:-1]*arr[1:-1] - arr[:-2]*arr[2:]
    # correct the data in the extremities
    tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]
    return(tkeo)

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
        absArray = maxValue - np.abs(np.arange(1000) - maxIndex) * step
        return np.where(absArray > 0, absArray, 0)

    cancelledWF = waveform
    wfArgmax = np.array([])
    integrate = np.sum(cancelledWF)
    init_integrate = integrate
    points_more_than_threshold = np.sum(cancelledWF > 0)
    init_points = points_more_than_threshold
    first = True
    noise_uncancelled_region = np.array([], dtype=int)
    
    while (np.max(cancelledWF) > 8 or integrate >= 80 or first) \
            and (points_more_than_threshold > 12 or integrate >= 96 or first) \
            and (wfArgmax.shape[0] or init_points >= 4 or init_integrate >= 60):
        argmax = np.argmax(cancelledWF)
        wfArgmax = np.append(wfArgmax, argmax)
        cancelledWF = cancelledWF - getCancel(argmax, 12)
        noise_uncancelled_region = np.intersect1d(
            np.union1d(noise_uncancelled_region, np.arange(argmax-8, argmax+9)),
            np.arange(1000)
        )
        judge_noise = cancelledWF[noise_uncancelled_region]
        judge_noise = np.where(judge_noise < 8, 0, judge_noise)
        cancelledWF[noise_uncancelled_region] = judge_noise
        points_more_than_threshold = cancelledWF.nonzero()[0].shape[0]
        integrate = np.sum(cancelledWF)
        first = False

    filteredWFArgmax = wfArgmax[np.all([wfArgmax >= 50, wfArgmax <= 600], axis=0)]
    return wfArgmax.shape[0], np.nanmean(filteredWFArgmax)


def getCancel(maxIndex, maxValue):
    '''
    getCancel
    '''
    step = maxValue / 8
    absArray = maxValue - np.abs(np.arange(1000) - maxIndex) * step
    return np.where(absArray > 0, absArray, 0)

def saveData(X, Y, path):
    h5 = h5py.File(path,'w')
    dataX = h5.create_dataset(name='X', data=X)
    dataY = h5.create_dataset(name='Y', data=Y)
    h5.close()


def saveans(ans, path):
    h5 = h5py.File(path,'w')
    A = h5.create_dataset(name='Answer', shape=(4000, ), dtype=np.dtype([('EventID', 'i'), ('p', 'f')]))
    A["EventID"] = np.array(range(4000))
    A["p"] = ans
    h5.close()
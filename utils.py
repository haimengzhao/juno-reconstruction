import pandas as pd
from torch import nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import joblib
from sklearn import preprocessing
from torch.utils.data import Dataset
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
    loss = (y-t)**2/t
    return "custom", np.mean(loss), False

def lossfunc_train(y, data):
    t = data.get_label()
    grad = 2*(y-t)/t
    hess = 2*np.ones_like(y)/t
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
        
        # 读取训练后的数据
        if datatype=='pro':
            X = torch.Tensor(np.array(file['X']))
            Y = torch.Tensor(np.array(file['Y']))
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
    eventIDs, indices = np.unique(dataset[type], return_index=True)
    indices = np.append(indices, dataset.shape[0]))
    num = np.diff(indices)
    return num, indices

def getCancel(maxIndex, maxValue):
    '''
    getCancel
    '''
    step = maxValue / 8
    absArray = maxValue - np.abs(np.arange(1000) - maxIndex) * step
    return np.where(absArray > 0, absArray, 0)

class TrainData(Dataset):
    def __init__(self, folder_path):
        self.X, self.Y = loadData(folder_path, 'pro')

    def __getitem__(self, index):
        return {'X':self.X[index], 'Y':self.Y[index]}

    def __len__(self):
        return len(self.Y)

def saveData(X, Y, path):
    h5 = h5py.File(path,'w')
    dataX = h5.create_dataset(name='X', data=X)
    dataY = h5.create_dataset(name='Y', data=Y)
    h5.close()

class Bottleneck(nn.Module):
    def __init__(self, planes):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu3(out)

        return out

def res_layers(planes, num_blocks, ):
    return nn.Sequential(*[Bottleneck(planes) for i in range(num_blocks)])

def conv_downsample(in_planes, out_planes):
    return nn.Sequential(
        nn.ReflectionPad1d(2),
        nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=2, bias=False),
        nn.BatchNorm1d(out_planes),
        nn.PReLU(),
    )

def conv_same(in_planes, out_planes):
    return nn.Sequential(
        nn.ReflectionPad1d(1),
        nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, bias=False),
        nn.BatchNorm1d(out_planes),
        nn.PReLU(),
    )

def saveans(ans, path):
    h5 = h5py.File(path,'w')
    A = h5.create_dataset(name='Answer', shape=(4000, ), dtype=np.dtype([('EventID', 'i'), ('p', 'f')]))
    A["EventID"] = np.array(range(4000))
    A["p"] = ans
    h5.close()
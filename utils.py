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
    file = h5py.File(datapath, "r")
    if datatype!='pro':
        print("Structure of data:")
        for key in file.keys():
            print(file[key], key, file[key].name)

    if datatype=='geo':
        return np.array(np.array(file['Geometry']).tolist())
    if datatype=='PT':
        PT = np.array(np.array(file['ParticleTruth']).tolist())
        PET = np.array(np.array(file['PETruth']).tolist())
        WF = np.array(file['Waveform'])
        return PET, WF, PT
    if datatype=='pro':
        X = torch.Tensor(np.array(file['X']))
        Y = torch.Tensor(np.array(file['Y']))
        return X, Y
    # PET = np.array(np.array(file['PETruth']).tolist())
    WF = np.array(file['Waveform'])
    return WF

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
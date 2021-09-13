import numpy as np
from utils import loadData
from utils import loadData, saveData, wavelet_denoising
from tqdm import tqdm

testpath = "../data/final-problem.h5"
trainpathroot = "../data/final-"
geopath = "../data/geonew.h5"
savetestpath = './data/testv9.h5'
savetrainpath = './data/trainv9.h5'

# hyperparam

thres = 300

if __name__=="__main__":

    # # testset
    print('start processing testset')
    testWF = loadData(testpath, 'test')
    print('testset loaded')

    w_test, j_test = np.unique(testWF['EventID'], return_index=True)
    j_test = np.append(j_test, len(testWF))
    numPEWtest = np.diff(j_test)
    invTestWF = 1000-testWF['Waveform']

    intWFtest = np.array([])
    intWFdRtest = np.array([])
    intWFdR2test = np.array([])
    numFilteredTest = np.array([])
    for arr in tqdm(np.split(invTestWF, j_test[1:-1])):

        # arr = wavelet_denoising(arr)

        head = np.mean(arr[:, :100], axis=1)
        tail = np.mean(arr[:, -100:], axis=1)
        base = np.maximum(head, tail)
        intpWF = np.sum(arr, axis=1)-base*1000
        filtered = np.maximum(intpWF, thres)
        numFiltered = np.sum(filtered==thres)
        intWFtest = np.append(intWFtest, np.sum(filtered)-numFiltered*thres)
        maxWF = np.argmax(arr, axis=1).reshape(-1,1) + 1
        # filtR = filtered*maxWF/250000*(1-(filtered==thres))
        filtdR = filtered/maxWF*(1-(filtered==thres))
        # intWFRtest = np.append(intWFRtest, np.sum(filtR))
        # intWFR2test = np.append(intWFR2test, np.sum(filtR*maxWF))
        intWFdRtest = np.append(intWFdRtest, np.sum(filtdR))
        intWFdR2test = np.append(intWFdR2test, np.sum(filtdR/maxWF))


    # Xtest = np.hstack((numPEWtest.reshape(-1, 1), intWFtest.reshape(-1, 1), intWFRtest.reshape(-1, 1), intWFR2test.reshape(-1, 1)))
    Xtest = np.hstack((numPEWtest.reshape(-1, 1), intWFtest.reshape(-1, 1), intWFdRtest.reshape(-1, 1), intWFdR2test.reshape(-1,1)))
    saveData(Xtest, np.array([0]), savetestpath)
    print('testset shape: ', Xtest.shape)

    # trainset
    print('start processing trainset')
    for i in tqdm(range(10)):
        trainPET, trainWF, trainPT = loadData(trainpathroot+str(i+2)+'.h5', 'PT')
        print('trainset '+str(i+2)+' loaded')

        e_tru, i_tru = np.unique(trainPET[:, 0], return_index=True)
        i_tru = np.append(i_tru, len(trainPET))
        w_tru, j_tru = np.unique(trainWF['EventID'], return_index=True)
        j_tru = np.append(j_tru, len(trainWF))
        numPET = np.diff(i_tru)
        numPEW = np.diff(j_tru)
        invTrainWF = 1000-trainWF['Waveform']

        intWF = np.array([])
        # intWFR = np.array([])
        # intWFR2 = np.array([])
        intWFdR = np.array([])
        intWFdR2 = np.array([])
        numFiltered = np.array([])
        for arr in tqdm(np.split(invTrainWF, j_tru[1:-1])):

            # arr = wavelet_denoising(arr)

            head = np.mean(arr[:, :100], axis=1)
            tail = np.mean(arr[:, -100:], axis=1)
            base = np.maximum(head, tail)
            intpWF = np.sum(arr, axis=1)-base*1000
            filtered = np.maximum(intpWF, thres)
            numFiltered = np.sum(filtered==thres)
            intWF = np.append(intWF, np.sum(filtered)-numFiltered*thres)
            maxWF = np.argmax(arr, axis=1).reshape(-1,1) + 1
            # filtR = filtered*maxWF/250000*(1-(filtered==thres))
            filtdR = filtered/maxWF*(1-(filtered==thres))
            # intWFR = np.append(intWFR, np.sum(filtR))
            # intWFR2 = np.append(intWFR2, np.sum(filtR*maxWF))
            intWFdR = np.append(intWFdR, np.sum(filtdR))
            intWFdR2 = np.append(intWFdR2, np.sum(filtdR/maxWF))

        if i>0:
            # X = np.vstack((X, np.hstack((numPEW.reshape(-1, 1), intWF.reshape(-1, 1), intWFR.reshape(-1, 1), intWFR2.reshape(-1, 1)))))
            X = np.vstack((X, np.hstack((numPEW.reshape(-1, 1), intWF.reshape(-1, 1), intWFdR.reshape(-1, 1), intWFdR2.reshape(-1,1)))))
            Y = np.vstack((Y, np.hstack((numPET.reshape(-1, 1), trainPT))))
        else:
            # X = np.hstack((numPEW.reshape(-1, 1), intWF.reshape(-1, 1), intWFR.reshape(-1, 1), intWFR2.reshape(-1, 1)))
            X = np.hstack((numPEW.reshape(-1, 1), intWF.reshape(-1, 1), intWFdR.reshape(-1, 1), intWFdR2.reshape(-1,1)))
            Y = np.hstack((numPET.reshape(-1, 1), trainPT))
        
        #saveData(X, Y, './data/trainv6-'+str(i+2)+'.h5')

    saveData(X, Y, savetrainpath)
    print('trainset shape: ', X.shape, Y.shape)
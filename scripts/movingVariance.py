

import numpy as np
import matplotlib.pyplot as plt 
# funciton to compute the moving standard deviation for a given window size
def movingStd(data, winSize):
    # compute a moving standard deviation
    stdVec = np.zeros((len(data)-winSize,1))
    for i in range(0,len(stdVec)):
        sqsum = 0
        meanVal = np.mean(data[i:i+winSize])
        for j in range(i,i+winSize):
            sqsum += (data[j]-meanVal)**2
        stdVec[i] = np.sqrt(sqsum/(winSize))
    
    # normalize
    stdVec = stdVec/max(stdVec)
    stdVec = stdVec.reshape(len(stdVec),)
    # print(repr(stdVec))
    # plt.plot(stdVec)
    # plt.show()
    return stdVec
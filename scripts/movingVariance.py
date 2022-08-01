# Copyright 2022 Agricultural-Robotics-Bonn
# All rights reserved.
#
# Software License Agreement (BSD 2-Clause Simplified License)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import numpy as np
from scipy.signal import find_peaks


# function to compute the moving standard deviation for a given window size

def movingStd(data, winSize=5):
    _data = data.copy()
    # compute a moving standard deviation
    stdVec = np.zeros((len(_data) - winSize, 1))
    for i in range(0, len(stdVec)):
        sqsum = 0
        meanVal = np.mean(_data[i:i + winSize])
        for j in range(i, i + winSize):
            sqsum += (_data[j] - meanVal) ** 2
        stdVec[i] = np.sqrt(sqsum / (winSize))

    # normalize
    stdVec = stdVec / max(stdVec)
    stdVec = stdVec.reshape(len(stdVec), )
    # print(repr(stdVec))
    # plt.plot(stdVec)
    # plt.show()
    return stdVec


def findPicksTroughths(movVarM, prominence=0.5, height=None):
    _movVarM = movVarM.copy()
    # find negative peaks (=least std, most stable line)
    peaksNeg, _ = find_peaks(-_movVarM, height=height)
    # find positive peaks (=the least stable lines = crop row transition)
    peaksPos, _ = find_peaks(_movVarM, prominence=0.5, height=height)
    return peaksPos, peaksNeg

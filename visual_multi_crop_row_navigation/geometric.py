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


def computeTheta(lineStart, lineEnd):
    """function to compute theta
    Args:
        lineStart (_type_): start point of line
        lineEnd (_type_): end point of line
    Returns:
        _type_: angle of line
    """
    return -(np.arctan2(abs(lineStart[1] - lineEnd[1]), lineStart[0] - lineEnd[0]))


def lineIntersectY(m, b, y):
    """ function to evaluate the estimated line
    Args:
        m (_type_): slope
        b (_type_): bias
        y (_type_): Y loc
    Returns:
        _type_: X loc
    """
    # line calculations
    x = m * y + b
    return x


def isInBox(box, p):
    """checks if point is inside the box
    Args:
        box (_type_): box
        p (_type_): point
    Returns:
        _type_: True or False
    """
    bl = box[0]
    tr = box[2]
    if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]):
        return True
    else:
        return False


def getImgLineUpDown(line, imageHeight):
    up = [line[1], 0]
    down = [line[0], imageHeight]
    return up, down


def lineIntersectImgSides(m, b, imageWidth):
    """_summary_
    Args:
        m (_type_): slope
        b (_type_): bias
    Returns:
        _type_: left and right interceptions
    """
    l_i = -b / m
    r_i = (imageWidth - b) / m
    return l_i, r_i


def lineIntersectImgUpDown(m, b, imageHeight):
    """function to compute the bottom and top intersect between the line and the image 
    Args:
        m (_type_): slope
        b (_type_): bias
    Returns:
        _type_: top and bottom intersection points on image boarders
    """
    # line calculations
    b_i = b
    t_i = m * imageHeight + b
    return t_i, b_i


def lineIntersectWin(m, b, imageHeight, topOffset, bottomOffset):
    """function to compute the bottom and top intersect between the line and the window
    Args:
        m (_type_): slope
        b (_type_): bias
    Returns:
        _type_: to and bottom intersection with a box
    """
    # line calculations
    b_i = m * bottomOffset + b
    t_i = m * (imageHeight - topOffset) + b
    return t_i, b_i


def getLineRphi(xyCords):
    """sets r , phi line 
    Args:
        xyCords (_type_): x, y coordinates of point
    Returns:
        _type_: r, phi of line
    """
    x_coords, y_coords = zip(*xyCords)
    coefficients = np.polyfit(x_coords, y_coords, 1)
    return coefficients[0], coefficients[1]

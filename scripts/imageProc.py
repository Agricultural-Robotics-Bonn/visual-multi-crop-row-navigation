#!/usr/bin/env python3

import scipy
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import itertools
from cv_bridge import CvBridge

from contours import *
from geometric import *
from featureMatching import *
from movingVariance import *


class imageProc:
    def __init__(self, scannerParams, contourParams, roiParams, featureParams):
        """# class FeatureExtractor to extract the the line Features: bottom point, angle

        Args:
            windowProp (_type_): _description_
            roiParams (_type_): _description_
            featureParams (_type_): _description_
        """
        self.roiParams = roiParams
        self.contourParams = contourParams
        self.scannerParams = scannerParams
        self.featureParams = featureParams

        self.reset()

    def reset(self):
        self.count = 0
        self.pointsInTop = 0
        self.pointsInBottom = 0

        # parameters
        self.bushy = False
        self.trackedLines = []
        self.windowPoints = []
        self.primaryRGBImg = []
        self.windowLocations = []
        self.cropLaneFound = False
        self.isInitialized = False

        # features
        self.rowAngles = []
        self.P = np.array([])
        self.allLineEnd = []
        self.allLineStart = []
        self.mainLine_up = np.array([])
        self.mainLine_down = np.array([])

        # window parameters
        self.trackingWindowOffsTop = 0
        self.trackingWindowOffsBottom = 0
        self.trackingWindowTopScaleRatio = 0.4  # scales the top of the window

    def findCropRows(self, rgbImg, depthImg, mode='RGB-D', bushy=False):
        """finds Crops Rows in the image based on RGB and depth data
        Args:
            bushy (bool, optional): _description_. Defaults to False.
        """
        self.primaryRGBImg = rgbImg.copy()
        self.primaryDepthImg = depthImg.copy()
        self.imgHeight, self.imgWidth, self.imgChannels = rgbImg.shape
        # if crop canopy type is busshy like Coriander
        self.bushy = bushy
        # check if current image is not Empty or None
        if self.primaryRGBImg is None or len(self.primaryRGBImg) == 0:
            print("#[ERR] CR-Scanner - Image is None, Check Topics or the Sensor !")
        else:
            print("#[INF] CR-Scanner - Searching for crop rows, please wait...")
            # initial RGB image process to get mask GreenIDx and plant centers
            self.mask, self.greenIDX, self.plantObjects, self.plantCenters = self.processRGBImage()
            self.lines2D, self.linesROIs2D = self.findCropRows2D(self.primaryRGBImg)
            self.lines3D, self.linesROIs3D = self.findCropRows3D(self.mask, self.plantCenters2D, self.primaryDepthImg)
            # merge both predictions to get more robust results!
        
        return self.cropLaneFound

    def findCropRows3D(self, maskImg, plantCenters2D, depthImg):
        lines = []
        linesROIs = []
        # project the plant centers to 3D

        # cluster points 

        # fit line on each cluster

        # project back 3D points to image

        # return the resutls
        return lines, linesROIs

    def findCropRows2D(self, rgbImg):
        # search for lines at given window locations using LQ-Adjustment
        lines, linesROIs, _ = self.findLinesInImage()
        # if any feature is observed during the scan
        if len(lines) != 0:
            np.set_printoptions(suppress=True)
            # compute moving standard deviation
            mvSignal = movingStd(lines[:, 0])
            # find positive an negative peaks of the signal
            peaksPos, peaksNeg = findPicksTroughths(mvSignal, 0.5)
            # locate best lines
            paramsBest, self.windowLocations = self.findCropRowsInMVSignal(peaksPos, peaksNeg,
                                                        mvSignal, lines, linesROIs)
            # if there is any proper line to follow
            if len(paramsBest) != 0:
                # get intersections of the lines with the image borders and
                # average these to get the main line
                avgLine = np.mean(np.c_[lineIntersectImgUpDown(
                    paramsBest[:, 1], paramsBest[:, 0], self.imgHeight)], axis=0)
                self.trackedLines = paramsBest
                self.top_x = avgLine[0]
                self.bottom_x = avgLine[1]
                #  get AvgLine in image cords
                self.mainLine_up, self.mainLine_down = getImgLineUpDown(
                    avgLine, self.imgHeight)
                # main features
                self.P = self.cameraToImage(
                    [self.bottom_x, self.imgHeight])
                self.ang = computeTheta(
                    self.mainLine_up, self.mainLine_down)
                # set parameters indicating a successfull initialization
                self.isInitialized = True
                self.cropLaneFound = True
                print('#[INF] Controller Initialized - Crop Rows:',
                        len(paramsBest),
                        ', Window positions:',
                        self.windowLocations.tolist())
            else:
                print(
                    '--- Initialisation failed - No lines detected due to peak detection ---')
                self.cropLaneFound = False
        else:
            print(
                '--- Initialisation failed - No lines detected by sweeping window ---')
            self.cropLaneFound = False
        
        return lines, linesROIs

    def findCropRowsInMVSignal(self, peaksPos, peaksNeg, mvSignal, lines, linesROIs):
        # if multiple lines
        if len(peaksPos) != 0 and len(peaksNeg) != 0 and len(mvSignal) != 0:
            # best line one each side of the crop row transition
            qualifiedLines = np.zeros((len(peaksPos)+1, 2))
            windowLocations = np.zeros((len(peaksPos)+1, 1))
            # get lines at each side of a positive peak (= crop row
            # transition and select the best)
            try:
                lidx = 0
                # every peak stands for 2 croprows
                for k in range(0, len(peaksPos) + 1):
                    # first peak
                    if k == 0:
                        lines = peaksNeg[peaksNeg < peaksPos[k]]
                    else:
                        # second to last-1
                        if k < len(peaksPos):
                            linestmp = peaksNeg[peaksNeg < peaksPos[k]]
                            lines = linestmp[linestmp > peaksPos[k-1]]
                        # last peak
                        else:
                            lines = peaksNeg[peaksNeg > peaksPos[k-1]]

                    if len(lines) != 0:
                        # best line (measured by variance)
                        bestLine = np.where(
                            mvSignal[lines] == np.min(mvSignal[lines]))[0][0]
                        # parameters of that line
                        qualifiedLines[lidx, :] = lines[lines[bestLine]]
                        # location of the window which led to this line
                        if linesROIs[lines[bestLine]] != [0]:
                            windowLocations[lidx,
                                                 :] = linesROIs[lines[bestLine]]
                            lidx += 1
            except:
                # fallback: just take all negative peaks
                qualifiedLines = lines[peaksNeg]
                windowLocations = np.mean(linesROIs[peaksNeg])

        # if there are no positive peaks but negative ones: no crop
        # row transition -> there might be just one line
        elif len(peaksPos) == 0 and len(peaksNeg) != 0 and len(mvSignal) != 0:
            lines = peaksNeg
            bestLine = np.where(mvSignal[lines] == np.min(mvSignal[lines]))[0][0]

            qualifiedLines = np.zeros((1, 2))
            windowLocations = np.zeros((1, 1))

            qualifiedLines[0, :] = lines[lines[bestLine]]
            windowLocations[0] = linesROIs[lines[bestLine]]
        else:
            qualifiedLines = []
            windowLocations = []

        return qualifiedLines,  windowLocations

    def findLinesInImage(self):
        """function to get lines at given window locations 
        Args:
            scannerWindows (_type_): _description_
        Returns:
            _type_: lines in image
        """
        if not self.initialized :
            # steps create linspace
            scannerWindows = np.linspace(self.scannerParams["scanStartPoint"],
                                         self.scannerParams["scanEndPoint"], 
                                         self.scannerParams["scanSteps"])
        # initialization
        lines = np.zeros((len(scannerWindows), 2))
        trackingWindows = np.zeros((len(scannerWindows), 1))
        meanLinesInWindows = np.zeros((len(scannerWindows), 1))
        self.windowPoints = np.zeros((len(scannerWindows), 6))
        angleVariance = None
        # counting the points in the top and bottom half
        self.pointsInBottom = 0
        self.pointsInTop = 0
        # for all windows
        for i in range(len(scannerWindows)):
            # define window
            if self.isInitialized and len(self.trackedLines) != 0:
                # get x values, where the top and bottom of the window intersect
                # with the previous found line
                win_intersect = lineIntersectWin(self.trackedLines[i, 1],
                                                 self.trackedLines[i, 0],
                                                 self.imgHeight,
                                                 self.offsT,
                                                 self.offsB)
                # window corner points are left and right of these
                xWinBL = win_intersect[0] - self.winSize/2
                xWinBR = xWinBL + self.winSize
                xWinTL = win_intersect[1] - self.winSize/2 * self.winRatio
                xWinTR = xWinTL + self.winSize * self.winRatio
                # y untouched (like normal rectangular)
                yWinB = self.offsB
                yWinT = self.imgHeight-self.offsT
                # store the corner points
                self.windowPoints[i, :] = [xWinBL, xWinTL, xWinTR,
                                           xWinBR, self.imgHeight-yWinB, self.imgHeight-yWinT]
            else:
                # initial window positions
                # window is centered around scannerWindows[i]
                xWinBL = scannerWindows[i]
                xWinBR = scannerWindows[i] + self.winSize
                xWinTL = xWinBL
                xWinTR = xWinBR
                yWinB = self.offsB
                yWinT = self.imgHeight-self.offsT
                # store the corner points
                self.windowPoints[i, :] = [xWinBL, xWinTL, xWinTR,
                                           xWinBR, self.imgHeight-yWinB, self.imgHeight-yWinT]

            plantsInCropRow = np.zeros((len(x), 2))
            for m in range(0, len(x)):
                # checking #points in upper/lower half of the image
                if y[m] < self.imgHeight/2:
                    self.pointsInTop += 1
                else:
                    self.pointsInBottom += 1
                # different query needed, if the window is a parallelogram
                if self.isInitialized and len(self.trackedLines) != 0 and self.trackedLines is not None:
                    # get the x value of the tracked line of the previous step
                    # at the y value of the query point
                    lXAtY = lineIntersectY(
                        self.trackedLines[i, 1], self.trackedLines[i, 0], y[m])
                    # window height
                    winH = yWinT-yWinB
                    # window width
                    winW = self.winSize - y[m]/winH * \
                        self.winSize*(1-self.winRatio)
                    # set up x constraint for query point
                    lXAtY_min = lXAtY - winW/2
                    lXAtY_max = lXAtY + winW/2
                    # test if the query point is inside the window
                    ptInWin = x[m] > lXAtY_min and x[m] < lXAtY_max and y[m] > yWinB and y[m] < yWinT
                # rectangular window
                else:
                    ptInWin = x[m] > xWinBL and x[m] < xWinBR and y[m] > yWinB and y[m] < yWinT

                # if the point is inside the window, add it to plantsInCropRow
                if ptInWin:
                    plantsInCropRow[m, :] = self.plantCentersInImage[:, m]

            # remove zero rows
            plantsInCropRow = plantsInCropRow[~np.all(
                plantsInCropRow == 0, axis=1)]
            print(len(plantsInCropRow))

            # line fit
            if len(plantsInCropRow) > 2:
                # flipped points
                ptsFlip = np.flip(plantsInCropRow, axis=1)
                # get line at scanning window
                xM, xB = getLineRphi(ptsFlip)
                t_i, b_i = lineIntersectImgUpDown(xM, xB, self.imgHeight)
                l_i, r_i = lineIntersectImgSides(xM, xB, self.imgHeight)
                # if the linefit does not return None and the line-image intersections
                # are within the image bounds
                if xM is not None and b_i >= 0 and b_i <= self.imgWidth:
                    lines[i, :] = [xB, xM]
                    # store the window location, which generated these line
                    # parameters
                    if self.isInitialized == False:
                        trackingWindows[i] = scannerWindows[i]
                    # mean x value of all points used for fitting the line
                    meanLinesInWindows[i] = np.median(plantsInCropRow, axis=0)[0]

        # if the feature extractor is initalized, adjust the window
        # size with the variance of the line fit
        if self.isInitialized and angleVariance is not None:
            self.winSize = max(3*angleVariance, self.winMinWidth)
            # print(self.winSize)

        # delete zero rows from line parameters
        lines = lines[~np.all(lines == 0, axis=1)]
        trackingWindows = trackingWindows[~np.all(trackingWindows == 0, axis=1)]
        meanLinesInWindows = meanLinesInWindows[~np.all(meanLinesInWindows == 0, axis=1)]

        return lines, trackingWindows, meanLinesInWindows

    def updateCropRowWindows(self):
        """function to get lines at given window locations 
        Returns:
            _type_: _description_
        """
        self.rowAngles = []
        # if the feature extractor is initalized
        if len(self.windowLocations) != 0:
            # get the lines at windows defined through previous found lines
            lines, _, meanLinesInWindows = self.getLinesInImage(self.windowLocations)
            # if 'all' lines are found by 'self.getLinesInImage'
            if len(lines) >= len(self.windowLocations):
                # location is always the left side of the window
                self.windowLocations = meanLinesInWindows - self.winSize/2
                # the line parameters are the new tracked lines (for the next step)
                self.trackedLines = lines
                # average image intersections of all found lines
                avgLine = np.mean(np.c_[lineIntersectImgUpDown(
                    lines[:, 1], lines[:, 0], self.imgHeight)], axis=0)
                # get intersection points
                self.top_x = avgLine[0]
                self.bottom_x = avgLine[1]
                #  get AvgLine in image cords
                self.mainLine_up, self.mainLine_down = getImgLineUpDown(
                    avgLine, self.imgHeight)
                # compute all intersections between the image and each line
                allLineIntersect = np.c_[lineIntersectWin(lines[:, 1],
                                                          lines[:, 0],
                                                          self.imgHeight,
                                                          self.offsT,
                                                          self.offsB)]
                # store start and end points of lines - mainly for plotting
                self.allLineStart = np.c_[
                    allLineIntersect[:, 0], self.imgHeight - self.offsB * np.ones((len(lines), 1))]
                self.allLineEnd = np.c_[
                    allLineIntersect[:, 1], self.offsT * np.ones((len(lines), 1))]

                for line in range(len(self.allLineEnd)):
                    ang = computeTheta(
                        self.allLineStart[line], self.allLineEnd[line])
                    # print(ang, self.allLineStart[line], self.allLineEnd[line])
                    self.rowAngles.append(ang)

                # main features
                self.P = self.cameraToImage([self.bottom_x, self.imgHeight])
                self.ang = computeTheta(self.mainLine_up, self.mainLine_down)

                # print('Tracking lines at window positions', self.windowLocations.tolist())
                self.cropLaneFound = True
            # if there are less lines than window positions
            else:
                print("Lost at least one line")
                self.cropLaneFound = False
        else:
            print('Running .initialize() first!')
            self.initialize()
        return self.cropLaneFound, self.mask

    def processRGBImage(self, rgbImg):
        """function to extract the greenidx and the contour center points
        Returns:
            _type_: _description_
        """
        # change datatype to enable negative values for self.greenIDX calculation
        rgbImg = rgbImg.astype('int32')

        rgbImg = self.applyROI(rgbImg)

        # cv.imshow("RGB image",imgInt32.astype('uint8'))
        # self.handleKey()
        #  get green index and binary mask (Otsu is used for extracting the green)
        mask, greenIDX = self.getExgMask(rgbImg)
        # find contours
        plantObjects = getPlantMasks(
            mask, self.min_contour_area, bushy=False)
        # get center of contours
        contCenterPTS = getCCenter(plantObjects)

        x = contCenterPTS[:, 1]
        y = contCenterPTS[:, 0]
        contCenterPTS = np.array([x, y])

        return mask, greenIDX, plantObjects, contCenterPTS

    def getExgMask(self, img):
        """Function to compute the green index of the image
        """
        _img = img.copy()

        _img = _img.astype('int32')
        # Vegetation Mask
        r = _img[:, :, 0]
        g = _img[:, :, 1]
        b = _img[:, :, 2]

        # calculate Excess Green Index and filter out negative values
        greenIDX = 2*g - r - b
        greenIDX[greenIDX < 0] = 0
        greenIDX = greenIDX.astype('uint8')

        # Otsu's thresholding after gaussian smoothing
        blur = cv.GaussianBlur(greenIDX, (5, 5), 0)
        threshold, threshIMG = cv.threshold(
            blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        # dilation
        kernel = np.ones((10, 10), np.uint8)
        binaryMask = cv.dilate(threshIMG, kernel, iterations=1)

        # Erotion
        # er_kernel = np.ones((10,10),dtype=np.uint8) # this must be tuned
        # binaryMask= cv.erode(binaryMask, er_kernel)

        return binaryMask, greenIDX

    def applyROI(self, img):
        _img = img.copy()
        # defining ROI windown on the image
        if self.roiParams["enable_roi"]:
            r_pts = [self.roiParams["p1"], self.roiParams["p2"],
                     self.roiParams["p3"], self.roiParams["p4"]]
            l_pts = [self.roiParams["p5"], self.roiParams["p6"],
                     self.roiParams["p7"], self.roiParams["p8"]]

            cv.fillPoly(_img, np.array([r_pts]), (0, 0, 0))
            cv.fillPoly(_img, np.array([l_pts]), (0, 0, 0))
        return _img

    def cameraToImage(self, P):
        """function to transform the feature point from camera to image frame
        Args:
            P (_type_): point in camera Fr
        Returns:
            _type_: point in image Fr
        """
        P[0] = P[0] - self.imgWidth/2
        P[1] = P[1] - self.imgHeight/2
        return P

    def drawGraphics(self):
        """function to draw the lines and the windows onto the image (self.primaryRGBImg)
        """
        # main line
        cv.line(self.primaryRGBImg, (int(self.mainLine_up[0]), int(self.mainLine_up[1])), (int(
            self.mainLine_down[0]), int(self.mainLine_down[1])), (255, 0, 0), thickness=3)
        # contoures
        cv.drawContours(self.primaryRGBImg,
                        self.plantObjects, -1, (10, 50, 150), 3)
        for i in range(0, len(self.allLineStart)):
            # helper lines
            cv.line(self.primaryRGBImg, (int(self.allLineStart[i, 0]), int(self.allLineStart[i, 1])), (int(
                self.allLineEnd[i, 0]), int(self.allLineEnd[i, 1])), (0, 255, 0), thickness=1)

            if self.isInitialized:
                #
                winBL = np.array(
                    [self.windowPoints[i, 0], self.windowPoints[i, -2]], np.int32)
                winTL = np.array(
                    [self.windowPoints[i, 1], self.windowPoints[i, -1]], np.int32)
                winTR = np.array(
                    [self.windowPoints[i, 2], self.windowPoints[i, -1]], np.int32)
                winBR = np.array(
                    [self.windowPoints[i, 3], self.windowPoints[i, -2]], np.int32)

                ptsPoly = np.c_[winBL, winBL, winBR, winTR, winTL].T
                cv.polylines(self.primaryRGBImg, [ptsPoly], True, (0, 255, 255))

        for i in range(len(self.plantCentersInImage[0])):
            # draw point on countur centers
            x = int(self.plantCentersInImage[0, i])
            y = int(self.plantCentersInImage[1, i])
            self.primaryRGBImg = cv.circle(
                self.primaryRGBImg, (x, y), 3, (255, 0, 255), 5)

    def handleKey(self, sleepTime=0):
        key = cv.waitKey(sleepTime)
        # Close program with keyboard 'q'
        if key == ord('q'):
            cv.destroyAllWindows()
            exit(1)

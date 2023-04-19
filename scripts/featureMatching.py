
import cv2 as cv
import numpy as np
import itertools
import matplotlib.pyplot as plt
from movingVariance import *

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class featureMatching:
    def __init__(self, featureParams):
        self.featureParams = featureParams
        self.simDistVec= []
        self.meanVec = []
        self.count  = 0
    
    def sampleCropRowFeatures(self, mode, rgbImg, greenIDx, binaryMask, wLocs):
        _rgbImg = rgbImg.copy()
        # get the correct window depending on the current mode
        rowId = None
        if mode == 3:
            rowId = 0
        else:
            rowId = -1
        self.refWindowLoc = wLocs[rowId]
        # int_coords = lambda x: np.array(x).round().astype(np.int32)
        # exterior = np.array([int_coords(wLocs[rowId].exterior.coords)])
        # bboxImg = self.cropBboxFromImage(_greenIDx, exterior)
        # get masked rgb Image (ExG masked)
        rgbMasked = self.maskRgb(rgbImg, binaryMask)
        # detect keypoints in image
        self.refKeypoints, self.refDescriptors = self.detectTrackingFeatures(rgbMasked)
        # filter out features in desired window location
        self.refKeypoints, self.refDescriptors = self.filterKeypoints(self.refKeypoints, 
                                                                      self.refDescriptors, 
                                                                      self.refWindowLoc)
        # draw keypoints on rgb Image
        self.drawKeyPoints(_rgbImg, self.refKeypoints, [255,0,0])
    
    def reset(self):
        pass

    def detectNewCropLane(self, mode, rgbImg, greenIDx, binaryMask, wLocs, numofCropRows):
        _rgbImg = rgbImg.copy()
        # get masked rgb Image (ExG masked)
        rgbMasked = self.maskRgb(rgbImg, binaryMask)
        # extract keypoints and descriptors from new detected line
        srcKeypoints, srcDescriptors = self.detectTrackingFeatures(rgbMasked)
        # filter in desired window
        windowLoc = self.refWindowLoc
        # filter out features in desired window location
        srcKeypoints, srcDescriptors = self.filterKeypoints(srcKeypoints, srcDescriptors, windowLoc)
        # find matches between ref and src keypoints
        matches = self.matchTrackingFeatures(srcKeypoints, srcDescriptors)
        # if not that indicates that the window is between two lines
        # draw keypoints on rgb Image
        self.drawKeyPoints(_rgbImg, srcKeypoints, [0,255,0])
        # collect matches
        self.simDistVec = list(self.simDistVec)
        self.simDistVec.append(len(matches))
        # create recognition signal
        peaks, troughs= [], []   
        if self.count > 90: 
            self.simDistVec = np.where(np.array(self.simDistVec) >= 10, 10, np.min(self.simDistVec))
            # compute moving standard deviation
            mvSignal = movingStd(self.simDistVec)
            # find positive an negative peaks of the signal
            peaks, troughs = findPicksTroughths(self.simDistVec, 0.5)
            print("peaks", len(peaks), "troughs", len(troughs))
             
        self.count+=1
        if ((len(peaks) >= numofCropRows and len(troughs) >= numofCropRows)):
            plt.plot(self.simDistVec)
            pickPoses = [self.simDistVec[p] for p in peaks]
            troughPoses = [self.simDistVec[p] for p in troughs]
            plt.plot(peaks, pickPoses , "o")
            plt.plot(troughs, troughPoses , "x")
            plt.plot(np.zeros_like(self.simDistVec), "--", color="gray")
            plt.show() 
            
        return (len(peaks) >= numofCropRows and len(troughs) >= numofCropRows)

    def maskRgb(self, rgbImg, mask):
        maskedRgb = np.zeros(np.shape(rgbImg), np.uint8)
        idx = (mask!=0)
        maskedRgb[idx] = rgbImg[idx]
        return maskedRgb

    def cropBboxFromImage(self, image, bbox):
        imgWHeight, imgWidth = np.shape(image)
        oneRowMask = np.zeros((imgWHeight, imgWidth), dtype=np.uint8)
        cv.fillPoly(oneRowMask, bbox, (255))
        res = cv.bitwise_and(image, image ,mask=oneRowMask)
        bbox = cv.boundingRect(bbox) # returns (x,y,w,h) of the rect
        bboxImg = res[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
        # cv.imshow("test", bboxImg)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return bboxImg

    def filterKeypoints(self, keyPoints, descriptors, polygon):
        filteredKeypoints = []
        filteredDescriptors = []
        for kp, des in itertools.izip(keyPoints, descriptors):
            if polygon.contains(Point(kp.pt[0], kp.pt[1])):
                filteredKeypoints.append(kp)    
                filteredDescriptors.append(des)
        return filteredKeypoints, filteredDescriptors
            
    def drawKeyPoints(self, rgbImg, keyPoints, color=[0, 0, 255], imShow=False):
        for kp in keyPoints:
            ptX = kp.pt[0]
            ptY = kp.pt[1]
            rgbImg[int(ptY)-3:int(ptY)+3, int(ptX) -
                            3:int(ptX)+3] = color
        if imShow:
            cv.imshow("test", rgbImg)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return rgbImg

    def detectTrackingFeatures(self, greenIDX):
        """ Function to compute the features of the line which has to be recognized

        Args:
            rgbImg (_type_): _description_
            greenIDX (_type_): _description_
            mode (_type_): _description_
            wLocs (_type_): _description_
            turnWindowWidth (_type_): _description_
            min_matching_dif_features (_type_): _description_
            max_matching_dif_features (_type_): _description_

        Returns:
            _type_: _description_
        """
        print("[bold blue]#[INF][/] detect Tracking Features")
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # get sift key points
        keyPointsRaw, descriptorsRaw = sift.detectAndCompute(greenIDX, None)
        matchingKeypoints = []
        featureDescriptors = []
        # maintain the keypoints lying in the first window
        for kp, desc in itertools.izip(keyPointsRaw, descriptorsRaw):
            matchingKeypoints.append(kp)
            featureDescriptors.append(desc)
        return matchingKeypoints, featureDescriptors
    
    def matchTrackingFeatures(self, srcKeypoints, srcDescriptors):
        """Function to compute the features in the second window
        """
        # Check if there's a suitable number of key points
        qualifiedMatches = []
        if len(srcKeypoints) > self.featureParams["minKeypointNum"]:
            # compute the matches between the key points
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(np.asarray(
                self.refDescriptors, np.float32), np.asarray(srcDescriptors, np.float32), k=2)
            # search for good matches (Lowes Ratio Test)
            for m, n in matches:
                if m.distance < 0.8*n.distance:
                    qualifiedMatches.append(m)
        else:
            print('Not enough key points for matching for matching')
   
        return qualifiedMatches
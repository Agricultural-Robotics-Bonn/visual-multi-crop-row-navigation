
import cv2 as cv
from matplotlib.image import BboxImage
import numpy as np
import itertools
import copy
from cv_bridge import CvBridge

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class featureMatching:
    def __init__(self, featureParams):
        self.featureParams = featureParams
    
    def sampleCropRowFeatures(self, mode, rgbImg, greenIDx, mask, wLocs):
        _greenIDx = greenIDx.copy()
        _rgbImg = rgbImg.copy()
        _rgbImgMasked = np.zeros(np.shape(_rgbImg), np.uint8)

        # get the correct window depending on the current mode
        rowId = None
        if mode == 3:
            rowId = 0
        else:
            rowId = -1
        # int_coords = lambda x: np.array(x).round().astype(np.int32)
        # exterior = np.array([int_coords(wLocs[rowId].exterior.coords)])
        # bboxImg = self.cropBboxFromImage(_greenIDx, exterior)
        idx = (mask!=0)
        _rgbImgMasked[idx] = _rgbImg[idx]
        self.refKeypoints, self.refDescriptors = self.detectTrackingFeatures(_rgbImgMasked)

        self.drawKeyPoints(_rgbImg, self.refKeypoints)

    def detectNewCropRow(self, mode, greenIDx, mask, wLocs):
        # find new crop rows and compare to old ones!
        # self.matchTrackingFeatures()
        return False, None

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

    def drawKeyPoints(self, rgbImg, keyPoints):
        for kp in keyPoints:
            ptX = kp.pt[0]
            ptY = kp.pt[1]
            rgbImg[int(ptY)-3:int(ptY)+3, int(ptX) -
                            3:int(ptX)+3] = [0, 0, 255]
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
        print("#[INF] detect Tracking Features")
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
                    
        print(len(matchingKeypoints), " Key Points in the first window were detected")
        return matchingKeypoints, featureDescriptors

    def matchTrackingFeatures(self, greenIDX, refDescriptors):
        """Function to compute the features in the second window
        """
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # get sift key points
        keyPointsRaw, descriptorsRaw = sift.detectAndCompute(greenIDX, None)
        keyPtsCurr = []
        descriptorsCurr = []
        # get the correct window depending on the current mode
        if mode == 3:
            windowLoc = self.wLocs[1]
        else:
            windowLoc = self.wLocs[-2]
        # maintain the keypoints lying in the second window
        for kp, desc in itertools.izip(keyPointsRaw, descriptorsRaw):
            ptX = kp.pt[0]
            ptY = kp.pt[1]
            # if the computed keypoint is in the second window keep it
            if ptX > (windowLoc - self.turnWindowWidth) and ptX < (windowLoc + self.turnWindowWidth):
                if ptY > self.max_matching_dif_features and ptY < (self.imgWidth - self.min_matching_dif_features):
                    keyPtsCurr.append(kp)
                    descriptorsCurr.append(desc)
                    # plot the key Points in the current image
                    self.primaryImg[int(ptY)-3:int(ptY)+3,
                                    int(ptX)-3:int(ptX)+3] = [255, 0, 0]

        # Check if there's a suitable number of key points
        good = []
        if len(keyPtsCurr) > self.matching_keypoints_th:
            # compute the matches between the key points
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(np.asarray(
                refDescriptors, np.float32), np.asarray(descriptorsCurr, np.float32), k=2)

            # search for good matches (Lowes Ratio Test)
            for m, n in matches:
                if m.distance < 0.8*n.distance:
                    good.append(m)

        # if not that indicates that the window is between two lines
        else:
            print('Not enough key points for matching for the ',
                self.nrNoPlantsSeen, ' time')
            self.nrNoPlantsSeen += 1
            print('# no plants seen:', self.nrNoPlantsSeen)
            if self.nrNoPlantsSeen > self.smoothSize:
                self.noPlantsSeen = True
        # cv bridge
        self.bridge = CvBridge()
        # publish processed image
        rosIMG = self.bridge.cv2_to_imgmsg(self.primaryImg, encoding='rgb8')
        self.numVec[self.count] = len(good)
        self.count += 1
        if self.count > self.smoothSize:
            self.meanVec[self.count] = sum(
                self.numVec[self.count-(self.smoothSize+1):self.count])/self.smoothSize

        # if the smoothed mean is descending the first row has passed the second window
        if self.meanVec[self.count] < self.meanVec[self.count-1] and self.meanVec[self.count] > self.matching_keypoints_th and self.noPlantsSeen:
            self.cnt += 1
            if self.cnt >= self.matching_keypoints_th:
                self.newDetectedRows += 1
                # if enough rows are passed
                if self.newDetectedRows == self.featureParams["linesToPass"]:
                    self.cnt = 0
                    print("All rows passed")
                    return True, rosIMG
                else:
                    print(self.newDetectedRows, " row(s) passed")
                    self.cnt = 0
                    # compute the new features in the new first window
                    self.detectTrackingFeatures()
                    return False, rosIMG
            else:
                print("No row passed, continuing")
                return False, rosIMG
        else:
            print("No row passed, continuing")
            return False, rosIMG

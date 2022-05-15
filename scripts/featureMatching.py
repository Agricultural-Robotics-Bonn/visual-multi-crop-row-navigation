
import cv2 as cv
import numpy as np
import itertools
from cv_bridge import CvBridge

class featureMatching:
    def __init__(self, featureParams):
        self.featureParams = featureParams

    def detectTrackingFeatures(self, mode, rgbImg, greenIDX, mask, windowLocations, turnWindowWidth):
        """ Function to compute the features of the line which has to be recognized

        Args:
            rgbImg (_type_): _description_
            greenIDX (_type_): _description_
            mode (_type_): _description_
            windowLocations (_type_): _description_
            turnWindowWidth (_type_): _description_
            min_matching_dif_features (_type_): _description_
            max_matching_dif_features (_type_): _description_

        Returns:
            _type_: _description_
        """
        print("#[INF] detect Tracking Features")
        self.imgWHeight, self.self.imgWidth, imgChannels = np.size(greenIDX)
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # get sift key points
        keyPointsRaw, descriptorsRaw = sift.detectAndCompute(greenIDX, None)
        matchingKeypoints = []
        featureDescriptors = []
        # get the correct window depending on the current mode
        if mode == 3:
            windowLoc = windowLocations[0]
        else:
            windowLoc = windowLocations[-1]
        # maintain the keypoints lying in the first window
        for kp, desc in itertools.izip(keyPointsRaw, descriptorsRaw):
            ptX = kp.pt[0]
            ptY = kp.pt[1]
            # if the computed keypoint is in the first window keep it
            # TODO check points in box with function!
            if ptX > (windowLoc - 2 * self.turnWindowWidth) and ptX < (windowLoc + 2 * self.turnWindowWidth):
                if ptY > self.featureParams["maxMatchingDifference"] and ptY < (self.imgWHeight - self.featureParams["minMatchingDifference"]):
                    matchingKeypoints.append(kp)
                    featureDescriptors.append(desc)
                    # plot the first key points
                    rgbImg[int(ptY)-3:int(ptY)+3, int(ptX) -
                        3:int(ptX)+3] = [0, 0, 255]

        print(len(matchingKeypoints), " Key Points in the first window were detected")
        return rgbImg, matchingKeypoints, featureDescriptors


    def matchTrackingFeatures(self, mode, greenIDX, matchingKeypoints, featureDescriptors):
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
            windowLoc = self.windowLocations[1]
        else:
            windowLoc = self.windowLocations[-2]
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
                featureDescriptors, np.float32), np.asarray(descriptorsCurr, np.float32), k=2)

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

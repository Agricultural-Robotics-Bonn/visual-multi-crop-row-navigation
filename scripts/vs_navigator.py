#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy


import math
import cv2 as cv
import Camera as cam
import featureExtractor as fex
import Controller as vs_controller
import numpy as np
import time

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import itertools


class vs_navigator:
    
    def __init__(self):
        # subscribed Topics (Images of front and back camera)
        front_topic = rospy.get_param('front_color_topic')
        back_topic = rospy.get_param('back_color_topic')
        self.sub_front_img = rospy.Subscriber(front_topic, Image, self.front_camera_callback, queue_size=1) 
        self.sub_back_img = rospy.Subscriber(back_topic, Image, self.back_camera_callback, queue_size=1) 

        # Initialize ros publisher, ros subscriber, topics we publish
        self.graphic_pub = rospy.Publisher('vs_nav/graphic',Image, queue_size=1)
        self.mask_pub = rospy.Publisher('vs_nav/mask',Image, queue_size=1)
        self.exg_pub = rospy.Publisher('vs_nav/ExG',Image, queue_size=1)

        cmd_vel_topic = rospy.get_param('cmd_vel_topic')
        self.velocity_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)

        # cv bridge
        self.bridge = CvBridge()

        self.img_resize_scale = rospy.get_param('img_resize_scale')

        # settings
        # Mode 1: Driving forward with front camera (starting mode)
        # Mode 2: Driving forward with back camera
        # Mode 3: Driving backwards with back camera
        # Mode 4: Driving backwards with front camera
        self.nav_mode = rospy.get_param('nav_mode')                   
        # debug mode without publishing velocities 
        self.stationary_debug = rospy.get_param('stationary_debug')
        #  speed limits
        self.angular_scaler = rospy.get_param('angular_scaler')
        self.max_linear_vel = rospy.get_param('max_linear_vel')
        self.min_linear_vel = rospy.get_param('min_linear_vel')
        self.max_angular_vel = rospy.get_param('max_angular_vel')
        self.min_angular_vel = rospy.get_param('min_angular_vel')
        # direction of motion 1: forward, -1:backward
        self.linear_motion_dir = 1
        #  used to recoder direction of motion
        self.angular_vel_buffer = list()
        # true: front camera, False: back camera
        self.primary_camera = True
        # switch process control parameters and settings
        self.switching_mode = False
        # counter of newly detected rows in switching process
        self.new_detected_rows = 0
        # buffer of detected keypoints to match for finding new rows
        self.matching_keypoints = []
        #  buffer of descriptor of detected keypoints
        self.feature_descriptors = []
        self.turnWindowWidth = 50
        self.linesToPass = rospy.get_param('lines_to_pass')            
        self.max_matching_dif_features = rospy.get_param('max_matching_dif_features')         
        self.min_matching_dif_features = rospy.get_param('min_matching_dif_features')           
        self.smoothSize = 5             # Parameter for controlling the smoothing effect
         # Threshold for keypoints
        self.matching_keypoints_th = rospy.get_param('matching_keypoints_th')
        # if there is no plant in the image
        self.noPlantsSeen = False
        self.nrNoPlantsSeen = 0
        self.numVec= np.zeros((300,1))
        self.meanVec = np.zeros((300,1))
        self.count = 0
        self.counter2 = 0


        self.windowProp = {
            "winSweepStart": rospy.get_param('winSweepStart'),
            "winSweepEnd": rospy.get_param('winSweepEnd'),
            "winMinWidth": rospy.get_param('winMinWidth'),
            "winSize": rospy.get_param('winSize')
        }
        #  in case of using bigger size image size, we suggest to set ROI 
        self.rioProp = {
            "p1": rospy.get_param('p1'),
            "p2": rospy.get_param('p2'),
            "p3": rospy.get_param('p3'),
            "p4": rospy.get_param('p4'),
            "p5": rospy.get_param('p5'),
            "p6": rospy.get_param('p6'),
            "p7": rospy.get_param('p7'),
            "p8": rospy.get_param('p8')
        }

        self.fexProp = {
            "min_contour_area": rospy.get_param('min_contour_area'),
            "max_coutour_height": rospy.get_param('max_coutour_height')
        }
        
        # images
        self.primary_img = []
        self.front_img = None
        self.back_img = None

        self.feature_extractor = fex.FeatureExtractor(self.windowProp, self.rioProp, self.fexProp)

        # camera
        self.camera = cam.Camera(1,1.2,0,1,np.deg2rad(-80),0.96,0,0,1)

        rospy.loginfo("#[VS] navigator initialied ... ")
   
    def getControls(self):
        """Function to control the robot   
        """

        msg = None
        
        # get the currently used image
        self.getCameraImage()
        
        # this is only False if the initialization in 'setImage' was unsuccessful
        if self.feature_extractor.initBool == False:
            rospy.logwarn("The initialization was unsuccessful!! ")
            # switch cameras
            self.switch_primary_camera()
            self.getCameraImage()
        else:  
            # if the robot is currently following a line and is not turning just compute the controls
            if not self.switching_mode:
                is_controls_computed, vel_msg, graphic_img = self.computeControls(self.primary_img, self.nav_mode)
                
                # if the is_controls_computed is False (no lines are found)
                if is_controls_computed == False:
                    rospy.logwarn("no lines are found !! is_controls_computed is False")
                    # switch to next mode
                    if not self.stationary_debug:
                        self.nav_mode += 1
                        
                    if self.nav_mode > 4:
                        self.nav_mode = 1
                    print("#[INF] Switched to mode ", self.nav_mode)
    
                    # if the mode is 2 or 4 one just switches the camera
                    if self.nav_mode == 2 or self.nav_mode == 4:
                        self.feature_extractor = fex.FeatureExtractor(self.windowProp, self.rioProp, self.fexProp)
                        self.switch_primary_camera()
                        self.getCameraImage()
                        self.getControls()
                    # if the mode is 1 or 3 the robot has to switch into new rows
                    else:
                        self.switching_mode = True
                        self.new_detected_rows = 0
                        self.count = 0
                        self.numVec= np.zeros((500,1))
                        self.meanVec = np.zeros((500,1))
                        print("#[INF] Turning Mode Enabled!!")
                        # Compute the features for the turning and stop the movement
                        self.computeTurnFeatures()
                        vel_msg = Twist()
                        vel_msg.linear.x = 0
                        
            # if the turning mode is enabled
            else: 
                # test if the condition for the row switching is fulfilled
                conditionSwitchFulfilled, graphic_img = self.testTurnFeatures()
                if conditionSwitchFulfilled:
                    # the turn is completed and the new lines to follow are computed
                    self.feature_extractor = fex.FeatureExtractor(self.windowProp, self.rioProp, self.fexProp)
                    self.switchDirection()
                    self.switching_mode = False
                    print("Turning Mode disabled, entering next rows")
                    vel_msg = Twist()
                    vel_msg.linear.y = 0
                    vel_msg.linear.x = 0.07 * self.linear_motion_dir
                else:
                    # if the condition is not fulfilled the robot moves continouisly sidewards
                    vel_msg = Twist()
                    vel_msg.linear.y = -0.08
        
        if not self.stationary_debug:
            # publish the commands to the robot
            if vel_msg is not None:
                self.velocity_pub.publish(vel_msg)
        else:
            print("#[INF]", vel_msg.linear.x, vel_msg.linear.y, vel_msg.angular.z)

        print("m:", self.nav_mode, "p-cam:", "front" if self.primary_camera else "back", 
              "vel-x,y,z",vel_msg.linear.x , vel_msg.linear.y, round(vel_msg.angular.z,3))

        # publish the modified image
        self.graphic_pub.publish(graphic_img)
        
    def testTurnFeatures(self):
        """Function to compute the features in the second window
        """

        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        
        # find the keypoints and feature_descriptors in the green index image with SIFT
        greenIDX = self.getGreenIDX()
        keyPointsRaw, descriptorsRaw = sift.detectAndCompute(greenIDX,None)
        keyPtsCurr = [];
        descriptorsCurr = [];
        
        # get the correct window depending on the current mode 
        if self.nav_mode == 3:
            windowLoc = self.feature_extractor.windowLocations[1]
        else:
            windowLoc = self.feature_extractor.windowLocations[-2]
            
        # maintain the keypoints lying in the second window
        for kp, desc in itertools.izip(keyPointsRaw, descriptorsRaw):
            ptX = kp.pt[0]
            ptY = kp.pt[1]
            # if the computed keypoint is in the second window keep it
            if ptX > (windowLoc - self.turnWindowWidth) and ptX < (windowLoc + self.turnWindowWidth):
               if ptY > self.max_matching_dif_features  and ptY < (self.feature_extractor.img_width - self.min_matching_dif_features):
                    keyPtsCurr.append(kp)
                    descriptorsCurr.append(desc)   
                    # plot the key Points in the current image
                    self.primary_img[int(ptY)-3:int(ptY)+3,int(ptX)-3:int(ptX)+3] = [255, 0, 0]
    
        # Check if there's a suitable number of key points 
        good = []
        if len(keyPtsCurr) > self.matching_keypoints_th:
            # compute the matches between the key points
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(np.asarray(self.feature_descriptors,np.float32),np.asarray(descriptorsCurr,np.float32),k=2)
                
            # search for good matches (Lowes Ratio Test) 
            for m,n in matches:
                if m.distance < 0.8*n.distance:
                    good.append(m)
            
        # if not that indicates that the window is between two lines
        else:
            print('Not enough key points for matching for the ',self.nrNoPlantsSeen, ' time')
            self.nrNoPlantsSeen += 1
            print('# no plants seen:',self.nrNoPlantsSeen)
            if self.nrNoPlantsSeen > self.smoothSize:
                self.noPlantsSeen = True
   
        # publish processed image
        rosIMG = self.bridge.cv2_to_imgmsg(self.primary_img, encoding='rgb8')
        self.numVec[self.count] = len(good)
        self.count += 1
        if self.count > self.smoothSize:
            self.meanVec[self.count] = sum(self.numVec[self.count-(self.smoothSize+1):self.count])/self.smoothSize

        # if the smoothed mean is descending the first row has passed the second window
        if self.meanVec[self.count]<self.meanVec[self.count-1] and self.meanVec[self.count] > self.matching_keypoints_th and self.noPlantsSeen:
            self.counter2 += 1
            if self.counter2 >= self.matching_keypoints_th:
                self.new_detected_rows += 1
                # if enough rows are passed
                if self.new_detected_rows == self.linesToPass:
                    self.counter2 = 0
                    print("All rows passed")
                    return True, rosIMG
                else: 
                    print(self.new_detected_rows, " row(s) passed")
                    self.counter2 = 0
                    # compute the new features in the new first window
                    self.computeTurnFeatures()
                    return False, rosIMG
            else:
                print("No row passed, continuing")
                return False, rosIMG
        else:
            print("No row passed, continuing")
            return False, rosIMG
        
    # Function to manage the switching between the two cameras
    def switch_primary_camera(self):
        self.primary_camera = not self.primary_camera

    # Function to manage the control variable for the driving direction
    def switchDirection(self):
        self.linear_motion_dir = -self.linear_motion_dir

    # Function to set the currently used image
    def getCameraImage(self):
        # The front image is used
        if self.primary_camera:
            self.primary_img = self.front_img;
        # The back image is used
        else:
            self.primary_img = self.back_img;

        self.feature_extractor.setImgProp(self.primary_img)    
        # If the feature extractor is not initialized yet, this has to be done
        if self.feature_extractor.initBool == False:
            self.feature_extractor.initialize()

    # Function to compute the controls when following a crop row
    def computeControls(self, img, mode):
        # extract features via the feature extractor
        mask = self.feature_extractor.updateLinesAtWindows()

        mask_msg = CvBridge().cv2_to_imgmsg(mask)
        mask_msg.header.stamp = rospy.Time.now()
        self.mask_pub.publish(mask_msg)

        exg_msg = CvBridge().cv2_to_imgmsg(self.feature_extractor.greenIDX)
        exg_msg.header.stamp = rospy.Time.now()
        self.exg_pub.publish(exg_msg)
        
        # get the processed image
        rosIMG = self.bridge.cv2_to_imgmsg(self.feature_extractor.processedIMG, encoding='rgb8')

        # initialize empty twist message
        vel_msg = Twist() 
    
        enoughPoints = True
        
        # Check if the end of the row is reached
        if mode == 1 or mode == 3: # driving forward
            if self.feature_extractor.pointsT < 2:
                enoughPoints = False 
        else: # driving backwards
            if self.feature_extractor.pointsB < 2:
                enoughPoints = False  
            scale = 1
            
        # if features are found and the end of the row is not reached yet
        if self.feature_extractor.lineFound and enoughPoints:
            sizeIMG = np.shape(img)
            
            # set linear speed and direction
            vel_msg.linear.x = 0.2 * self.linear_motion_dir
                
            # extract the features
            x = self.feature_extractor.P[0]
            y = self.feature_extractor.P[1]
            t = self.feature_extractor.ang
            
            if mode == 2 or mode == 4:
                x = self.feature_extractor.topIntersect-(sizeIMG[1]/2)
                self.rot_dir = -1
            else:
                self.rot_dir = 1

            # t *= self.rot_dir
            # define desired and actual feature vector
            des = np.array([0,sizeIMG[0]/2,0])
            act = np.array([x,y,t])

            # compute controls
            controls = vs_controller.Controller(self.camera,des,act,self.max_angular_vel)
            
            vel_msg.angular.z = self.angular_scaler * controls
            # print("controls",vel_msg.angular.z)
            turning_sign = vel_msg.angular.z / vel_msg.angular.z
            # vel_msg.angular.z = min(self.min_angular_vel, max(math.fabs(vel_msg.angular.z), self.max_angular_vel)) * turning_sign
            
            # store the command in a cache
            self.angular_vel_buffer.append(vel_msg.angular.z)
            
            return True, vel_msg, rosIMG
        
        # if no lines are found or the end of the row is reached
        else:
            if not enoughPoints:
                rospy.loginfo("Reached the end of the row!")
            else:
                rospy.loginfo("No lines found!")
            
            # using the last known control 
            if len(self.angular_vel_buffer) == 0:
                self.angular_vel_buffer.append(0.0)
            vel_msg.angular.z = self.angular_vel_buffer[-1]
            return False, vel_msg, rosIMG
    
    # Function to compute the features of the line which has to be recognized
    def computeTurnFeatures(self):
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        
        # find the keypoints and feature_descriptors in the green index image with SIFT
        greenIDX = self.getGreenIDX()
        
        keyPointsRaw, descriptorsRaw = sift.detectAndCompute(greenIDX,None)
        self.matching_keypoints = [];
        self.feature_descriptors = [];
        
        # get the correct window depending on the current mode
        if self.nav_mode == 3:
            windowLoc = self.feature_extractor.windowLocations[0]
        else:
            windowLoc = self.feature_extractor.windowLocations[-1]
            
        # maintain the keypoints lying in the first window
        for kp, desc in itertools.izip(keyPointsRaw, descriptorsRaw):
            ptX = kp.pt[0]
            ptY = kp.pt[1]
            
            # if the computed keypoint is in the first window keep it
            if ptX > (windowLoc - 2*self.turnWindowWidth) and ptX < (windowLoc + 2*self.turnWindowWidth):
               if ptY > self.max_matching_dif_features  and ptY < (self.feature_extractor.image_size[0] - self.min_matching_dif_features):
                    self.matching_keypoints.append(kp)
                    self.feature_descriptors.append(desc)
                    # plot the first key points 
                    self.primary_img[int(ptY)-3:int(ptY)+3,int(ptX)-3:int(ptX)+3] = [0, 0, 255]

        print(len(self.matching_keypoints)," Key Points in the first window were detected")
        self.noPlantsSeen = False
        self.nrNoPlantsSeen = 0
        self.counter2 = 0
            
    # Function to compute the green index of the image
    def getGreenIDX(self):
        imgInt32 = self.primary_img.astype('int32')
        
        # Vegetation Mask
        r = imgInt32[:,:,0]
        g = imgInt32[:,:,1]
        b = imgInt32[:,:,2]
    
        # calculate Excess Green Index and filter out negative values
        greenIDX = 2*g - r - b
        greenIDX[greenIDX<0] = 0
        greenIDX = greenIDX.astype('uint8')

        return greenIDX

    # Function to deal with the front image, called by the subscriber
    def front_camera_callback(self, data):
        # get and set new image from the ROS topic
        self.front_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        # if the image is not empty, called by the subscriber
        if self.front_img is not None and self.back_img is not None:
            # compute and publish robot controls if the image is currently used
            if self.primary_camera:
                self.getControls()
                     
    # Function to deal with the back image
    def back_camera_callback(self, data):
        # get and set new image from the ROS topic
        self.back_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        # if the image is not empty
        if self.front_img is not None and self.back_img is not None:
            # compute and publish robot controls if the image is currently used
            if not self.primary_camera:
                self.getControls()
    
    
if __name__ == '__main__':
    #  initializing vs navigator node 
    rospy.init_node('vs_navigator', anonymous=False)
    rospy.loginfo("#[VS] Visual_servoing navigator node running ...")
    
    # instantiating navigator class
    navigator = vs_navigator()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logwarn("#[VS] Shutting down Phenobot Controller")                    
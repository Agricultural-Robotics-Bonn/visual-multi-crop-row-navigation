#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from future.builtins import input

import rospy
import math
import cv2 as cv
import Camera as cam
import imageProc as imc
import controller as visualServoingCtl
import numpy as np
import time

from geometry_msgs.msg import Twist
import itertools

import message_filters

import tf2_ros
import tf_conversions
import tf2_geometry_msgs
import image_geometry
import geometry_msgs.msg
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

from featureMatching import *

class vs_nodeHandler:
    
    def __init__(self):
        # cv bridge
        self.bridge = CvBridge()
        self.queue_size = 1

        # subscribed Topics (Images of front and back camera)
        self.frontColor_topic = rospy.get_param('frontColor_topic')
        self.frontDepth_topic = rospy.get_param('frontDepth_topic')
        self.frontCameraInfo_topic = rospy.get_param('frontCameraInfo_topic')

        self.frontColor_sub = message_filters.Subscriber(
            self.frontColor_topic, Image, queue_size=self.queue_size)
        self.frontDepth_sub = message_filters.Subscriber(
            self.frontDepth_topic, Image, queue_size=self.queue_size)
        self.frontCameraInfo_sub = message_filters.Subscriber(
            self.frontCameraInfo_topic, CameraInfo, queue_size=self.queue_size)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.frontColor_sub, self.frontDepth_sub, self.frontCameraInfo_sub], queue_size=self.queue_size, slop=0.5)
        self.ts.registerCallback(self.frontSyncCallback)

        self.backColor_topic = rospy.get_param('backColor_topic')
        self.backDepth_topic = rospy.get_param('backDepth_topic')
        self.backCameraInfo_topic = rospy.get_param('backCameraInfo_topic')

        self.backColor_sub = message_filters.Subscriber(
            self.backColor_topic, Image, queue_size=self.queue_size)
        self.backDepth_sub = message_filters.Subscriber(
            self.backDepth_topic, Image, queue_size=self.queue_size)
        self.backCameraInfo_sub = message_filters.Subscriber(
            self.backCameraInfo_topic, CameraInfo, queue_size=self.queue_size)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.backColor_sub, self.backDepth_sub, self.backCameraInfo_sub], queue_size=self.queue_size, slop=0.5)
        self.ts.registerCallback(self.backSyncCallback)

        # Initialize ros publisher, ros subscriber, topics we publish
        self.graphic_pub = rospy.Publisher('vs_nav/graphic',Image, queue_size=1)
        self.mask_pub = rospy.Publisher('vs_nav/mask',Image, queue_size=1)
        self.exg_pub = rospy.Publisher('vs_nav/ExG',Image, queue_size=1)

        cmd_vel_topic = rospy.get_param('cmd_vel_topic')
        self.velocity_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)

        # settings
        # Mode 1: Driving forward with front camera (starting mode)
        # Mode 2: Driving forward with back camera
        # Mode 3: Driving backwards with back camera
        # Mode 4: Driving backwards with front camera
        self.navigationMode = rospy.get_param('navigationMode')                   
        # debug mode without publishing velocities 
        self.stationaryDebug = rospy.get_param('stationaryDebug')
        # rotation direction FLAG
        self.rotationDir = 1
        # direction of motion 1: forward, -1:backward
        if self.navigationMode == 1 or self.navigationMode == 2:
            self.linearMotionDir = 1
        else:
            self.linearMotionDir = -1
        # true: front camera, False: back camera
        if self.isUsingFrontCamera(): 
            self.primaryCamera = True
        else:
            self.primaryCamera = False
        # switch process control parameters and settings
        self.switchingMode = False
        #  used to recoder direction of motion
        self.omegaBuffer = list()

        self.scannerParams = {
            "scanSteps": rospy.get_param('scanSteps'),
            "scanEndPoint": rospy.get_param('scanEndPoint'),
            "scanStartPoint": rospy.get_param('scanStartPoint'),
            "scanWindowWidth": rospy.get_param('scanWindowWidth')
        }
        #  in case of using bigger size image size, we suggest to set ROI 
        self.rioParams = {
            "enable_roi": rospy.get_param('enable_roi'),
            "p1": rospy.get_param('p1'),
            "p2": rospy.get_param('p2'),
            "p3": rospy.get_param('p3'),
            "p4": rospy.get_param('p4'),
            "p5": rospy.get_param('p5'),
            "p6": rospy.get_param('p6'),
            "p7": rospy.get_param('p7'),
            "p8": rospy.get_param('p8')
        }

        self.contourParams = {
            "imgResizeRatio": rospy.get_param('imgResizeRatio'),
            "minContourArea": rospy.get_param('minContourArea'),
        }

        self.featureParams = {
            "linesToPass": rospy.get_param('linesToPass'),
            "matchingKeypointsNum": rospy.get_param('matchingKeypointsNum'),
            "maxMatchingDifference": rospy.get_param('maxMatchingDifference'),
            "minMatchingDifference": rospy.get_param('minMatchingDifference'),
        }

        # speed limits
        self.omegaScaler = rospy.get_param('omegaScaler')
        self.maxOmega = rospy.get_param('maxOmega')
        self.minOmega = rospy.get_param('minOmega')
        self.maxLinearVel = rospy.get_param('maxLinearVel')
        self.minLinearVel = rospy.get_param('minLinearVel')
        
        # images
        self.frontImg = None
        self.frontDepth = None
        self.backImg = None
        self.backDepth = None

        self.velocityMsg = Twist()
        self.enoughPoints = True

        # camera
        self.camera = cam.Camera(1,1.2,0,1,np.deg2rad(-80),0.96,0,0,1)
        self.imageProcessor = imc.imageProc(self.scannerParams, 
                                            self.contourParams, 
                                            self.rioProp, 
                                            self.featureParams)

        self.camera_model = image_geometry.PinholeCameraModel()
        rospy.loginfo('Detection Camera initialised, {}, , {}'.format(
            self.color_topic, self.depth_topic, self.camera_info_topic))
        print('')

        rospy.loginfo("#[VS] navigator initialied ... ")
        
    # main function to guide the robot through crop rows
    def navigate(self):
        # get the currently used image
        primaryRGB, primaryDepth = self.getProcessingImage(self.frontImg, self.backImg)
        # If the feature extractor is not initialized yet, this has to be done
        if not self.imageProcessor.findCropRows(primaryRGB, primaryDepth, mode='RGB-D'):
            # this is only False if the initialization in 'setImage' was unsuccessful
            rospy.logwarn("The initialization was unsuccessful!! ")
            # switch cameras
            primaryRGB, primaryDepth = self.getProcessingImage(self.frontImg, self.backImg, switchCamera=True)
        else:  
            # if the robot is currently following a line and is not turning just compute the controls
            if not self.switchingMode:
                validPathFound, ctlCommands = self.computeControls()
                # if the validPathFound is False (no lines are found)
                if not validPathFound:
                    rospy.logwarn("no lines are found !! validPathFound is False")
                    # switch to next mode
                    self.updateNavigationStage()
                    # if the mode is 2 or 4 one just switches the camera
                    if self.isExitingLane():
                        self.imageProcessor.reset()
                        self.getProcessingImage(self.frontImg, self.backImg, switchCamera=True)
                        self.navigate()
                    # if the mode is 1 or 3 the robot follows rows 
                    else:
                        print("#[INF] Turning Mode Enabled!!")
                        self.switchingMode = True
                        self.velocityMsg = Twist()
                        self.velocityMsg.linear.x = 0.0
                        self.velocityMsg.linear.y = 0.0
                        self.velocityMsg.angular.z = 0.0
                        time.sleep(1.0)
                        # Compute the features for the turning and stop the movement
                        detectTrackingFeatures(self.navigationMode,
                                               self.imageProcessor.primaryRGBImg,
                                               self.imageProcessor.greenIDX,
                                               self.imageProcessor.mask, 
                                               self.imageProcessor.windowLocations,
                                               turnWindowWidth = 100)
                        time.sleep(1.0)
                else:
                    self.velocityMsg.linear.x = ctlCommands[0]
                    self.velocityMsg.linear.y = 0.0
                    self.velocityMsg.angular.z = ctlCommands[1]
            # if the turning mode is enabled
            else: 
                # test if the condition for the row switching is fulfilled
                newLaneFound, graphic_img = matchTrackingFeatures(self.navigationMode)
                if newLaneFound:
                    # the turn is completed and the new lines to follow are computed
                    self.imageProcessor.reset()
                    self.switchDirection()
                    self.switchingMode = False
                    self.velocityMsg = Twist()
                    self.velocityMsg.linear.x = 0.07 * self.linearMotionDir
                    self.velocityMsg.linear.y = 0.0
                    self.velocityMsg.angular.z = 0.0
                    print("#[INF] Turning Mode disabled, entering next rows")
                else:
                    # if the condition is not fulfilled the robot moves continouisly sidewards
                    self.velocityMsg = Twist()
                    self.velocityMsg.linear.x = 0.0
                    self.velocityMsg.linear.y = -0.08
                    self.velocityMsg.angular.z = 0.0
                    # check Odometry for safty (not to drive so much!)
                    print("#[INF] Side motion to find New Lane ...")
        
        if not self.stationaryDebug:
            # publish the commands to the robot
            if self.velocityMsg is not None:
                self.velocity_pub.publish(self.velocityMsg)

        print("#[INF] m:", 
              self.navigationMode, 
              "p-cam:", "front" if self.primaryCamera else "back", 
              "vel-x,y,z",
              self.velocityMsg.linear.x, 
              self.velocityMsg.linear.y, 
              round(self.velocityMsg.angular.z, 3),
              self.linearMotionDir,
              self.rotationDir)

        # Publish the Graphics image
        self.imageProcessor.drawGraphics()
        graphic_img = self.bridge.cv2_to_imgmsg(self.imageProcessor.processedIMG, encoding='rgb8')
        self.graphic_pub.publish(graphic_img)
        # publish predicted Mask
        mask_msg = CvBridge().cv2_to_imgmsg(self.mask)
        mask_msg.header.stamp = rospy.Time.now()
        self.mask_pub.publish(mask_msg)
        # publish Exg image 
        exg_msg = CvBridge().cv2_to_imgmsg(self.imageProcessor.greenIDX)
        exg_msg.header.stamp = rospy.Time.now()
        self.exg_pub.publish(exg_msg)
    
    def frontSyncCallback(self, rgbImage, depthImage, camera_info_msg):
        self.camera_model.fromCameraInfo(camera_info_msg)
        # print("here")
        try:
            # Convert your ROS Image message to OpenCV2
            self.frontImg = self.bridge.imgmsg_to_cv2(rgbImage, "bgr8")
        except CvBridgeError as e:
            print(e)
        try:
            # Convert your ROS Image message to OpenCV2
            # The depth image is a single-channel float32 image
            # the values is the distance in mm in z axis
            cv_depth = self.bridge.imgmsg_to_cv2(depthImage, "passthrough")
            # Convert the depth image to a Numpy array since most cv functions
            # require Numpy arrays.
            self.frontDepth = np.array(cv_depth, dtype=np.float32)
            # Normalize the depth image to fall between 0 (black) and 1 (white)
            cv.normalize(self.frontDepth, self.frontDepth,
                          0.0, 1.0, cv.NORM_MINMAX)
        except CvBridgeError as e:
            print(e)

        # get image size
        self.imgHeight, self.imgWidth, self.imgCh = self.frontImg.shape
        # if the image is not empty
        if self.frontImg is not None and self.backImg is not None:
            # compute and publish robot controls if the image is currently used
            if self.primaryCamera:
                self.navigate()
    
    def backSyncCallback(self, rgbImage, depthImage, camera_info_msg):
        self.camera_model.fromCameraInfo(camera_info_msg)
        # print("here")
        try:
            # Convert your ROS Image message to OpenCV2
            self.backImg = self.bridge.imgmsg_to_cv2(rgbImage, "bgr8")
        except CvBridgeError as e:
            print(e)
        try:
            # Convert your ROS Image message to OpenCV2
            # The depth image is a single-channel float32 image
            # the values is the distance in mm in z axis
            cv_depth = self.bridge.imgmsg_to_cv2(depthImage, "passthrough")
            # Convert the depth image to a Numpy array since most cv functions
            # require Numpy arrays.
            self.backDepth = np.array(cv_depth, dtype=np.float32)
            # Normalize the depth image to fall between 0 (black) and 1 (white)
            cv.normalize(self.backDepth, self.backDepth,
                          0.0, 1.0, cv.NORM_MINMAX)
        except CvBridgeError as e:
            print(e)

        # get image size
        self.imgHeight, self.imgWidth, self.imgCh = self.frontImg.shape
        # if the image is not empty
        if self.frontImg is not None and self.backImg is not None:
            # compute and publish robot controls if the image is currently used
            if self.primaryCamera:
                self.navigate()

    def front_camera_callback(self, data):
        """Function to deal with the front image, called by the subscriber

        Args:
            data (_type_): _description_
        """
        # get and set new image from the ROS topic
        self.frontImg = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        # get image size
        self.imgHeight, self.imgWidth, self.imgCh = self.frontImg.shape
        # if the image is not empty
        if self.frontImg is not None and self.backImg is not None:
            # compute and publish robot controls if the image is currently used
            if self.primaryCamera:
                self.navigate()
                       
    def back_camera_callback(self, data):
        """Function to deal with the back image

        Args:
            data (_type_): _description_
        """
        # get and set new image from the ROS topic
        self.backImg = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        # get image size
        self.imgHeight, self.imgWidth, self.imgCh = self.backImg.shape
        # if the image is not empty
        if self.frontImg is not None and self.backImg is not None:
            # compute and publish robot controls if the image is currently used
            if not self.primaryCamera:
                self.navigate()

    def updateNavigationStage(self):
        """updates navigation stagte (one higher or reset to 1 from > 4)
        """
        self.navigationMode += 1
        if self.navigationMode > 4:
            self.navigationMode = 1  
        inputKey = input("#[INF] Press Enter to continue with mode:")
        print("#[INF] Switched to mode ", self.navigationMode)
    
    def isExitingLane(self):
        """condition of line existing action, modes 2, 4

        Returns:
            _type_: _description_
        """
        if self.navigationMode == 2 or self.navigationMode == 4:
            return True
        else:
            return False
    
    def isFollowingLane(self):
        """if following a lane, modes 1, 3

        Returns:
            _type_: _description_
        """
        if self.navigationMode == 1 or self.navigationMode == 3:
            return True
        else:
            return False
    
    def isUsingFrontCamera(self):
        if self.navigationMode == 1 or self.navigationMode == 4:
            return True
        else:
            return False
    
    def isUsingBackCamera(self):
        if self.navigationMode == 2 or self.navigationMode == 3:
            return True
        else: 
            return False

    def switchDirection(self):
        """Function to manage the control variable for the driving direction
        """
        self.linearMotionDir = -self.linearMotionDir
        print("#####################switched Direction of Motion ...", self.linearMotionDir)
    
    def switchRotationDir(self):
        """Function to manage the control variable for the driving rotation
        """
        self.rotationDir = -self.rotationDir
        print("&&&&&&&&&&&&&&&&&&&&&switched Direction of Rotation ...", self.rotationDir)

    def getProcessingImage(self, frontImg, backImg, switchCamera=False):
        """Function to set the currently used image

        Args:
            frontImg (_type_): _description_
            backImg (_type_): _description_
            switchCamera (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if switchCamera:
            print("switch camera to the other ...")
            # Function to manage the switching between the two cameras
            self.primaryCamera = not self.primaryCamera
        # The front image is used
        if self.primaryCamera:
            primaryImg = frontImg
        # The back image is used
        else:
            primaryImg = backImg
        return primaryImg     
    
    def computeControls(self):
        """Function to compute the controls when following a crop row

        Returns:
            _type_: _description_
        """
        # extract features via the feature extractor
        lineFound, self.mask = self.imageProcessor.updateCropRowWindows()
        # if features are found and the end of the row is not reached yet
        if lineFound: 
            # print("line found -- compute controls...")
            # extract the features
            x = self.imageProcessor.P[0]
            y = self.imageProcessor.P[1]
            t = self.imageProcessor.ang
            # define desired and actual feature vector
            desiredFeature = np.array([0.0, self.imgWidth/2, 0.0])
            actualFeature = np.array([x, y, t])
            # compute controls
            controls = visualServoingCtl.visualServoingCtl(self.camera,
                                                desiredFeature, 
                                                actualFeature, 
                                                self.maxLinearVel)
            if self.isExitingLane():
                self.rotationDir = -1
            else:
                self.rotationDir = 1
            # scale rotational velocity 
            omega = self.omegaScaler * controls 
            # set linear speed and direction
            rho = 0.2 * self.linearMotionDir
            # store the command in a cache
            self.omegaBuffer.append(omega)
            
            return True, [rho, omega]
        
        # if no lines are found or the end of the row is reached
        else:
            print("No line found -- End of Row")
            # using the last known control 
            if len(self.omegaBuffer) == 0:
                self.omegaBuffer.append(0.0)
            # straight exit
            omega = 0.0
            rho = 0.05 * self.linearMotionDir
            return False, [rho, omega]

    def transformTargets(self, targets, frameName):
        cameraToOdomTF = None
        targetList = []
        try:
            cameraToOdomTF = self.tfBuffer.lookup_transform(
                frameName, self.camera_model.tfFrame(), rospy.Time(0))
        except:
            rospy.logerr("lookup_transform " + frameName + " to " + self.camera_model.tfFrame() + " failed !!!")
        
        if not cameraToOdomTF == None:
            for i in range(len(targets)):
                stem = tf2_geometry_msgs.do_transform_pose(targets[i], cameraToOdomTF)
                targetList.append(stem)

        return targetList
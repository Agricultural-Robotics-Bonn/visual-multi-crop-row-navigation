#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


from __future__ import division, print_function

import time

from cv_bridge import CvBridge, CvBridgeError
from future.builtins import input
from geometry_msgs.msg import Twist
import image_geometry
import message_filters
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

import tf2_geometry_msgs.tf2_geometry_msgs
import visual_multi_crop_row_navigation.camera as cam
import visual_multi_crop_row_navigation.controller as visualServoingCtl
from visual_multi_crop_row_navigation.featureMatching import *
import visual_multi_crop_row_navigation.imageProc as imc


# import tf2_geometry_msgs


class VisualServoingNode(Node):

    def __init__(self):
        super().__init__('vs_navigator')
        self.get_logger().info('#[VS] Visual_servoing navigator node running ...')
        # cv bridge
        self.bridge = CvBridge()
        self.queue_size = 1

        # subscribed Topics (Images of front and back camera)
        self.frontColor_sub = message_filters.Subscriber(self, Image, 'front/color')
        self.frontDepth_sub = message_filters.Subscriber(self, Image, 'front/depth')
        self.frontCameraInfo_sub = message_filters.Subscriber(self, CameraInfo,
                                                              'front/camera_info')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.frontColor_sub, self.frontDepth_sub, self.frontCameraInfo_sub],
            queue_size=self.queue_size, slop=0.5)
        self.ts.registerCallback(self.frontSyncCallback)

        self.backColor_sub = message_filters.Subscriber(self, Image, 'back/color')
        self.backDepth_sub = message_filters.Subscriber(self, Image, 'back/depth')
        self.backCameraInfo_sub = message_filters.Subscriber(self, CameraInfo, 'back/camera_info')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.backColor_sub, self.backDepth_sub, self.backCameraInfo_sub],
            queue_size=self.queue_size, slop=0.5)
        self.ts.registerCallback(self.backSyncCallback)

        # Initialize ros publisher, ros subscriber, topics we publish
        self.graphic_pub = self.create_publisher(Image, 'vs_nav/graphic', qos_profile=1)
        self.mask_pub = self.create_publisher(Image, 'vs_nav/mask', qos_profile=1)
        self.exg_pub = self.create_publisher(Image, 'vs_nav/ExG', qos_profile=1)

        self.velocity_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile=1)

        # declare all parameters and their defaults
        self.declare_parameters('', [
            ('navigationMode', 1),
            ('stationaryDebug', False),
            ('scanSteps', 5),
            ('scanEndPoint', 1280),
            ('scanStartPoint', 0),
            ('scanWindowWidth', 128),
            ('enable_roi', True),
            ('p1', [0, 0]),
            ('p2', [350, 0]),
            ('p3', [10, 720]),
            ('p4', [0, 720]),
            ('p5', [830, 0]),
            ('p6', [1280, 0]),
            ('p7', [1280, 720]),
            ('p8', [1270, 720]),
            ('imgResizeRatio', 100),
            ('minContourArea', 10),
            ('linesToPass', 1),
            ('minKeypointNum', 10),
            ('maxMatchingDifference', 100),
            ('minMatchingDifference', 0),
            ('scaleRatio', 0.4),
            ('topOffset', 0),
            ('bottomOffset', 0),
            ('trackingBoxWidth', 200),
            ('omegaScaler', 0.1),
            ('maxOmega', 0.05),
            ('minOmega', 0.01),
            ('maxLinearVel', 0.5),
            ('minLinearVel', 0.01),
        ])

        # settings
        # Mode 1: Driving forward with front camera (starting mode)
        # Mode 2: Driving forward with back camera
        # Mode 3: Driving backwards with back camera
        # Mode 4: Driving backwards with front camera
        self.navigationMode = self.get_parameter('navigationMode').value
        # debug mode without publishing velocities
        self.stationaryDebug = self.get_parameter('stationaryDebug').value
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
        #  used to recoder direction of motion
        self.omegaBuffer = list()

        self.scannerParams = {
            'scanSteps': self.get_parameter('scanSteps').value,
            'scanEndPoint': self.get_parameter('scanEndPoint').value,
            'scanStartPoint': self.get_parameter('scanStartPoint').value,
            'scanWindowWidth': self.get_parameter('scanWindowWidth').value
        }

        #  in case of using bigger size image size, we suggest to set ROI
        self.rioParams = {
            'enable_roi': self.get_parameter('enable_roi').value,
            'p1': self.get_parameter('p1').value,
            'p2': self.get_parameter('p2').value,
            'p3': self.get_parameter('p3').value,
            'p4': self.get_parameter('p4').value,
            'p5': self.get_parameter('p5').value,
            'p6': self.get_parameter('p6').value,
            'p7': self.get_parameter('p7').value,
            'p8': self.get_parameter('p8').value
        }

        self.contourParams = {
            'imgResizeRatio': self.get_parameter('imgResizeRatio').value,
            'minContourArea': self.get_parameter('minContourArea').value,
        }

        self.featureParams = {
            'linesToPass': self.get_parameter('linesToPass').value,
            'minKeypointNum': self.get_parameter('minKeypointNum').value,
            'maxMatchingDifference': self.get_parameter('maxMatchingDifference').value,
            'minMatchingDifference': self.get_parameter('minMatchingDifference').value,
        }

        self.trackerParams = {
            "scaleRatio": self.get_parameter('scaleRatio').value,
            "topOffset": self.get_parameter('topOffset').value,
            "bottomOffset": self.get_parameter('bottomOffset').value,
            "trackingBoxWidth": self.get_parameter('trackingBoxWidth').value,
        }

        # speed limits
        self.omegaScaler = self.get_parameter('omegaScaler').value
        self.maxOmega = self.get_parameter('maxOmega').value
        self.minOmega = self.get_parameter('minOmega').value
        self.maxLinearVel = self.get_parameter('maxLinearVel').value
        self.minLinearVel = self.get_parameter('minLinearVel').value

        # images
        self.frontImg = None
        self.frontDepth = None
        self.backImg = None
        self.backDepth = None

        self.velocityMsg = Twist()
        self.enoughPoints = True
        self.samplingDone = False

        # camera
        self.camera = cam.Camera(1, 1.2, 0, 1, np.deg2rad(-80), 0.96, 0, 0, 1)
        self.imageProcessor = imc.imageProc(self.scannerParams,
                                            self.contourParams,
                                            self.rioParams,
                                            self.trackerParams)

        self.cameraModel = image_geometry.PinholeCameraModel()
        self.get_logger().info('Detection Camera initialised..')
        print('')

        self.featureMatcher = featureMatching(self.featureParams)

        self.get_logger().info("#[VS] navigator initialied ... ")

    # main function to guide the robot through crop rows
    def navigate(self):
        # get the currently used image
        primaryRGB, primaryDepth = self.getProcessingImage(self.frontImg,
                                                           self.frontDepth,
                                                           self.backImg,
                                                           self.backDepth)
        # If the feature extractor is not initialized yet, this has to be done
        if not self.imageProcessor.findCropLane(primaryRGB, primaryDepth, mode='RGB-D'):
            # this is only False if the initialization in 'setImage' was unsuccessful
            self.get_logger().warn("The initialization was unsuccessful!! ")
        else:
            print("cropLaneFound", self.imageProcessor.cropLaneFound, "cropRowEnd",
                  self.imageProcessor.cropRowEnd)
            # if the robot is currently following a line and is not turning just compute the controls
            if self.isFollowingLane():
                self.imageProcessor.trackCropLane(self.navigationMode)
                ctlCommands = self.computeControls(self.imageProcessor.cropLaneFound,
                                                   self.imageProcessor.P,
                                                   self.imageProcessor.ang)

                if not self.imageProcessor.cropRowEnd:
                    print("#[INF] Following detected Lane ...")
                    self.setRobotVelocities(ctlCommands[0], 0.0, ctlCommands[1])

                elif self.imageProcessor.cropRowEnd:
                    print("#[INF] End of Lane detected ...")

                    if self.isExitingLane():
                        self.updateNavigationStage()
                    else:
                        self.updateNavigationStage()
                        self.imageProcessor.reset()
                        self.imageProcessor.isInitialized = False
                        self.imageProcessor.cropRowEnd = False
                        self.switchCamera()

            elif self.isSwitchingLane() and not self.samplingDone:
                print("#[INF] Sampling the Lane!!")
                self.stopRobot(2.0)
                # Compute the features for the turning and stop the movement
                self.featureMatcher.sampleCropRowFeatures(self.navigationMode,
                                                          self.imageProcessor.primaryRGBImg,
                                                          self.imageProcessor.greenIDX,
                                                          self.imageProcessor.mask,
                                                          self.imageProcessor.rowTrackingBoxes)
                self.samplingDone = True
                self.stopRobot(2.0)
            else:
                # test if the condition for the row switching is fulfilled
                foundNewCropLane = self.featureMatcher.detectNewCropLane(self.navigationMode,
                                                                         self.imageProcessor.primaryRGBImg,
                                                                         self.imageProcessor.greenIDX,
                                                                         self.imageProcessor.mask,
                                                                         self.imageProcessor.rowTrackingBoxes,
                                                                         self.imageProcessor.numOfCropRows)
                if foundNewCropLane:
                    self.stopRobot(2.0)
                    print("following new Lane !!")
                    # the turn is completed and the new lines to follow are computed
                    self.switchDirection()
                    print("#[INF] Turning Mode disabled, Entering next lane")
                    self.updateNavigationStage()
                    self.imageProcessor.reset()
                    self.imageProcessor.isInitialized = False
                    self.imageProcessor.cropRowEnd = False
                else:
                    # if the condition is not fulfilled the robot moves continouisly sidewards
                    self.setRobotVelocities(0.0, -0.05, 0.0)
                    # check Odometry for safty (not to drive so much!)
                    print("#[INF] Side motion to find New Lane ...")

        self.publishImageTopics()

        print("#[INF] m:",
              self.navigationMode,
              "p-cam:", "front" if self.primaryCamera else "back",
              "vel-x,y,z",
              self.velocityMsg.linear.x,
              self.velocityMsg.linear.y,
              round(self.velocityMsg.angular.z, 3),
              self.imageProcessor.numOfCropRows)

    def publishImageTopics(self):
        # Publish the Graphics image
        self.imageProcessor.drawGraphics()
        graphic_img = self.bridge.cv2_to_imgmsg(self.imageProcessor.graphicsImg, encoding='bgr8')
        self.graphic_pub.publish(graphic_img)
        # publish predicted Mask
        mask_msg = CvBridge().cv2_to_imgmsg(self.imageProcessor.mask)
        mask_msg.header.stamp = self.get_clock().now().to_msg()
        self.mask_pub.publish(mask_msg)
        # publish Exg image 
        exg_msg = CvBridge().cv2_to_imgmsg(self.imageProcessor.greenIDX)
        exg_msg.header.stamp = self.get_clock().now().to_msg()
        self.exg_pub.publish(exg_msg)

    def setRobotVelocities(self, x, y, omega):
        self.velocityMsg = Twist()
        self.velocityMsg.linear.x = x
        self.velocityMsg.linear.y = y
        self.velocityMsg.angular.z = omega
        if not self.stationaryDebug:
            # publish the commands to the robot
            if self.velocityMsg is not None:
                self.velocity_pub.publish(self.velocityMsg)

    def stopRobot(self, delay):
        self.velocityMsg = Twist()
        self.velocityMsg.linear.x = 0.0
        self.velocityMsg.linear.y = 0.0
        self.velocityMsg.angular.z = 0.0
        time.sleep(delay)
        self.velocity_pub.publish(self.velocityMsg)

    def frontSyncCallback(self, rgbImage, depthImage, camera_info_msg):
        self.cameraModel.fromCameraInfo(camera_info_msg)
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
        self.cameraModel.fromCameraInfo(camera_info_msg)
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
        self.imgHeight, self.imgWidth, self.imgCh = self.backImg.shape
        # if the image is not empty
        if self.frontImg is not None and self.backImg is not None:
            # compute and publish robot controls if the image is currently used
            if not self.primaryCamera:
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
        if self.navigationMode > 6:
            self.navigationMode = 1
        inputKey = input("#[INF] Press Enter to continue with mode:")
        print("#[INF] Switched to mode ", self.navigationMode)

    def isSwitchingLane(self):
        """condition of line existing action, modes 3, 6

        Returns:
            _type_: _description_
        """
        if self.navigationMode == 3 or self.navigationMode == 6:
            return True
        else:
            return False

    def isExitingLane(self):
        """condition of line existing action, modes 2, 5

        Returns:
            _type_: _description_
        """
        if self.navigationMode in [2, 5]:
            return True
        else:
            return False

    def isFollowingLane(self):
        """if following a lane, modes 1, 4

        Returns:
            _type_: _description_
        """
        if self.navigationMode in [1, 2, 4, 5]:
            return True
        else:
            return False

    def isUsingFrontCamera(self):
        if self.navigationMode == 1 or self.navigationMode == 5 or self.navigationMode == 6:
            return True
        else:
            return False

    def isUsingBackCamera(self):
        if self.navigationMode == 2 or self.navigationMode == 3 or self.navigationMode == 4:
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

    def switchCamera(self):
        print("switch camera to the other ...")
        # Function to manage the switching between the two cameras
        self.primaryCamera = not self.primaryCamera

    def getProcessingImage(self, frontRgb, frontDepth, backRgb, backDepth):
        """Function to set the currently used image

        Args:
            frontImg (_type_): _description_
            backImg (_type_): _description_

        Returns:
            _type_: _description_
        """
        # The front image is used
        if self.primaryCamera:
            primaryRgb = frontRgb
            primaryDepth = frontDepth
        # The back image is used
        else:
            primaryRgb = backRgb
            primaryDepth = backDepth
        return primaryRgb, primaryDepth

    def computeControls(self, LaneFound, P, Angle):
        """Function to compute the controls when following a crop row

        Returns:
            _type_: _description_
        """
        if LaneFound:
            # define desired and actual feature vector
            desiredFeature = np.array([0.0, self.imgWidth / 2, 0.0])
            actualFeature = np.array([P[0], P[1], Angle])
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

            return [rho, omega]
        else:
            # using the last known control 
            if len(self.omegaBuffer) == 0:
                self.omegaBuffer.append(0.0)
            # straight exit
            omega = 0.0
            rho = 0.05 * self.linearMotionDir
            return [rho, omega]

    def transformTargets(self, targets, frameName):
        cameraToOdomTF = None
        targetList = []
        try:
            cameraToOdomTF = self.tfBuffer.lookup_transform(
                frameName, self.cameraModel.tfFrame(), Time(0))
        except:
            self.get_logger().error(
                "lookup_transform " + frameName + " to " + self.cameraModel.tfFrame() + " failed !!!")

        if not cameraToOdomTF == None:
            for i in range(len(targets)):
                # until this PR https://github.com/ros2/geometry2/pull/533 is merged,
                # we have to manually clone this package in the same workspace
                stem = tf2_geometry_msgs.do_transform_pose(targets[i], cameraToOdomTF)
                targetList.append(stem)

        return targetList

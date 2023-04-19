#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from vs_nodeHandler import vs_nodeHandler 

    
if __name__ == '__main__':
    #  initializing vs navigator node 
    rospy.init_node('vs_navigator', anonymous=False)
    rospy.loginfo("#[VS] Visual_servoing navigator node running ...")
    
    # instantiating navigator class
    nodeHandler = vs_nodeHandler()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logwarn("#[VS] Shutting down Phenobot Controller")
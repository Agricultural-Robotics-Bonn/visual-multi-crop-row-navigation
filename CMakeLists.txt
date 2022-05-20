cmake_minimum_required(VERSION 3.0.2)
project(visual_multi_crop_row_navigation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES msr_project
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
# catkin_install_python(PROGRAMS scripts/vs_nodeHandler.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

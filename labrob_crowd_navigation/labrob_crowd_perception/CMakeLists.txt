cmake_minimum_required(VERSION 3.0.2)
project(labrob_crowd_perception)
find_package(catkin REQUIRED
  rospy
  cv_bridge
  labrob_crowd_navigation_msgs
  message_filters
  pal_detection_msgs
  sensor_msgs
)
catkin_python_setup()
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES labrob_crowd_control
  CATKIN_DEPENDS
    rospy
    cv_bridge
    labrob_crowd_navigation_msgs
    message_filters
    pal_detection_msgs
    sensor_msgs
#  DEPENDS system_lib
)
include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)
install(PROGRAMS
  scripts/crowd_perception_ground_truth
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

cmake_minimum_required(VERSION 3.0.2)
project(labrob_crowd_navigation_utils)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  labrob_crowd_navigation_msgs
  rospy
)
catkin_python_setup()
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES labrob_crowd_navigation_utils
  CATKIN_DEPENDS
    geometry_msgs
    labrob_crowd_navigation_msgs
    rospy
#  DEPENDS system_lib
)
include_directories(
# include
#  ${catkin_INCLUDE_DIRS}
)

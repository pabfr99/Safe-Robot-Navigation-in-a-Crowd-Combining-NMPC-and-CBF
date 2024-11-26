cmake_minimum_required(VERSION 3.0.2)
project(labrob_crowd_control)
find_package(catkin REQUIRED
  rospy
  labrob_crowd_navigation_msgs
)
catkin_python_setup()
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES labrob_crowd_control
  CATKIN_DEPENDS
    rospy
    labrob_crowd_navigation_msgs
    labrob_crowd_navigation_utils
#  DEPENDS system_lib
)
include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)
install(PROGRAMS
  scripts/cbf_qp_motion_generation_manager
  scripts/nmpc_motion_generation_manager
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

cmake_minimum_required(VERSION 3.0.2)
project(labrob_crowd_navigation)
find_package(catkin REQUIRED COMPONENTS
  labrob_crowd_control
  labrob_crowd_navigation_msgs
  labrob_crowd_navigation_utils
  labrob_crowd_perception
)
catkin_metapackage()

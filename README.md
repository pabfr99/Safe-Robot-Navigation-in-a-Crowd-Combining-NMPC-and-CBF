# AMR23-FP4

## Installation
To make sure everything is working properly make sure you have Ubuntu 20.04 with
ROS Noetic. Install catkin_tools, create a catkin workspace and clone this
repository in the `src` folder. Make sure you also have
[pal_person_detector_opencv](https://github.com/pal-robotics/pal_person_detector_opencv)
and [pal_msgs](https://github.com/pal-robotics/pal_msgs/tree/indigo-devel) inside
the `src` folder. Compile in *Release* mode
by properly setting your catkin workspace:
```bash
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
```
Build your code by running the following command:
```bash
catkin build
```

## Usage
### Gazebo simulation
To run the Gazebo simulation:
```bash
roslaunch labrob_tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=WORLD
```
where `WORLD` is one of the worlds in one of the packages `labrob_gazebo_worlds`
or `pal_gazebo_worlds`.

### Crowd perception
To run the crowd perception module using ground truth data:
```bash
roslaunch labrob_crowd_perception crowd_perception_ground_truth.launch
```

### Motion generation
To run the motion generation module:
```bash
roslaunch labrob_crowd_control nmpc_motion_generation_manager.launch
```

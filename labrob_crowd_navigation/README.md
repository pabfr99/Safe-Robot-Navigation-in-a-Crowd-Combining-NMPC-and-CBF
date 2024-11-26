# labrob_crowd_navigation
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

To run the crowd perception module using only the RGB-D camera:
```bash
roslaunch labrob_crowd_perception crowd_perception_camera.launch
```

To run the crowd perception module using only the laser scan:
```bash
roslaunch labrob_crowd_perception crowd_perception_laser_scan.launch
```

### Motion generation
To run the motion generation module based on CBF-QP:
```bash
roslaunch labrob_crowd_control cbf_qp_motion_generation_manager.launch
```

To run the motion generation module based on NMPC:
```bash
roslaunch labrob_crowd_control nmpc_motion_generation_manager.launch
```
### Visualization
To visualize the results of the simulations:
```bash
python NMPCVisualizer.py -l <plots_folder/simulation>
```

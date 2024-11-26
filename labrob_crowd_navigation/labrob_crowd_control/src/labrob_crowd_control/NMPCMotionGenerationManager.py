import rospy
import numpy as np
import json
import re

from labrob_crowd_control.MotionGenerationManager import *
from labrob_crowd_control.NMPC import *
from labrob_crowd_control.NMPCVisualizer import *
from labrob_crowd_control.AreaSwitcher import *
from labrob_crowd_control.Interpolator import *

class NMPCMotionGenerationManager(MotionGenerationManager):
    def __init__(self, world_name):
        # extract the name of the world and define the parameters
        self.define_world(world_name=world_name, 
                          with_v_ref=True)
        MotionGenerationManager.__init__(self, world_name)
        # controller
        self.nmpc_controller = NMPC(world_name=self.world_name, 
                                    num_obstacles=self.num_obstacles, 
                                    with_v_ref=self.with_v_ref)
        # define the path and the area switcher
        self.define_path(max_vel_reduction_fact=2.0, 
                         threshold_to_goal=0.1)
        self.define_area_switcher(h_number=2)
        # initialize some arrays to store the data and usefull counters
        self.init_memory_and_counters()
    
    def define_world(self, 
                     world_name, 
                     with_v_ref=True):
        name = re.search(r'_([a-zA-Z]+)_(\d+)', world_name)
        self.world_name = name.group(1)
        self.num_obstacles = int(name.group(2))
        self.with_v_ref = with_v_ref
    
    def define_path(self, 
                    max_vel_reduction_fact=2.0, 
                    threshold_to_goal=0.1):
        self.ref_velocity = self.nmpc_controller.driving_vel_max/max_vel_reduction_fact
        self.threshold = threshold_to_goal
        self.interpolator = Interpolator(world_name=self.world_name, 
                                         avg_velocity=self.ref_velocity, 
                                         dt=self.dt)
        self.path = self.interpolator.get_path()
        self.waypoints = self.interpolator.waypoints
        
    def define_area_switcher(self, 
                             h_number=0):
        # Define the heuristic to use for switching between position constraints
        self.heuristics = ['prediction', 'next_waypoint', 'nearest_waypoint', 'time']
        self.heuristic = self.heuristics[h_number]
        # array of position constraint index
        self.switcher = AreaSwitcher(heuristic=self.heuristic, 
                                     nmpc_controller=self.nmpc_controller, 
                                     waypoints=self.waypoints,
                                     world_name=self.world_name)
        self.current_time = time.time()
        
    def init_memory_and_counters(self):
        self.max_length=20000
        self.failure_counter=0
        self.counter=2
        self.config_idx = 0 
        self.actual_config = np.zeros((self.max_length, 5))
        self.crowd_position = np.zeros((self.max_length, self.num_obstacles, 2))
        self.switch_index = np.zeros(self.max_length)
        self.position_error = np.zeros((self.max_length, 2))
        self.pred_motion = np.zeros((self.nmpc_controller.N, 2))
        self.driving_accelerations = np.zeros(self.max_length)
        self.steering_accelerations = np.zeros(self.max_length)
        self.control_inputs_nmpc = np.zeros((self.max_length, self.nmpc_controller.nu))
        self.update_time = np.zeros(self.max_length)
        self.failure_mem=np.zeros(self.max_length)
    
    def define_output_dir(self):
        script_dir = os.path.dirname(__file__)
        if self.with_v_ref:
            self.plot_dir = 'plots_v_cost'
            print("Plots with velocity reference in the cost function will be saved in the 'plots_v_cost' directory.")
        else:
            self.plot_dir = 'plots'
            print("Plots without velocity reference in the cost function will be saved in the 'plots' directory.")
        save_dir = f'{self.plot_dir}/{self.world_name}_{self.num_obstacles}_{self.heuristic}'
        out_dir = os.path.join(script_dir, '..', '..', save_dir)
        self.out_dir = os.path.normpath(out_dir)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
    def save_simulation_data(self):
        print("Saving simulation data...")
        self.define_output_dir()
        base_filename = os.path.join(self.out_dir, f"{self.world_name}_{self.num_obstacles}_{self.heuristic}")
        npz_filename = f"{base_filename}.npz"
        np.savez(npz_filename, 
                 actual_config=self.actual_config[:self.config_idx],
                 crowd_position=self.crowd_position[:self.config_idx],
                 position_error=self.position_error[:self.config_idx],
                 switch_index=self.switch_index[:self.config_idx],
                 control_inputs_nmpc=self.control_inputs_nmpc[:self.config_idx],
                 driving_accelerations=self.driving_accelerations[:self.config_idx],
                 steering_accelerations=self.steering_accelerations[:self.config_idx],
                 update_time=self.update_time[:self.config_idx],
                 path=self.path[:self.config_idx],
                 failure_mem=self.failure_mem[:self.config_idx])
        filtered_area = [[(point[0], point[1]) for point in area if not np.isnan(point[0])] for area in self.nmpc_controller.areas]
        other_data = {
            'waypoints': self.interpolator.waypoints.tolist(),
            'heuristic': self.heuristic,
            'world': self.world_name,
            'areas': filtered_area,
            'num_people': self.num_obstacles,
            'dt': self.dt,
        }
        
        json_filename = f"{base_filename}_other_data.json"
        with open(json_filename, 'w') as json_file:
            json.dump(other_data, json_file, indent=4)

        print(f"Simulation data saved in {npz_filename} and {json_filename}.")
        
    def update_reference(self, 
                         x, 
                         y):
        current_position = np.array([x, y])
        distances = np.linalg.norm(self.path[:, :2] - current_position, axis=1)
        nearest_idx = np.argmin(distances)    
        end_idx = min(nearest_idx + self.nmpc_controller.N + 1, len(self.path))
        # get the path segment that is closest to the robot
        path_segment = self.path[nearest_idx:end_idx]
        # if the path segment is shorter than the horizon, repeat the last waypoint
        if len(path_segment) < self.nmpc_controller.N + 1:
            last_waypoint = self.waypoints[-1]
            path_segment = np.vstack([path_segment, np.full((self.nmpc_controller.N + 1 - len(path_segment), 2), last_waypoint)])
        # create a reference trajectory with the x and y values of the path segment
        q_ref = np.zeros((self.nmpc_controller.nq, self.nmpc_controller.N + 1))
        q_ref[self.nmpc_controller.x_idx, :] = path_segment[:, 0]
        q_ref[self.nmpc_controller.y_idx, :] = path_segment[:, 1]
        if self.with_v_ref:
            q_ref[self.nmpc_controller.v_idx, :] = self.ref_velocity
        return q_ref
        
    def update(self):
        try:
            start_time = time.time()
            
            # Set state from configuration and wheel encoders:
            driving_velocity = self.nmpc_controller.wheel_radius * 0.5 * \
                    (self.right_wheel_angular_velocity_rt + self.left_wheel_angular_velocity_rt)
            steering_velocity = self.nmpc_controller.wheel_radius / self.nmpc_controller.wheel_separation * \
                    (self.right_wheel_angular_velocity_rt - self.left_wheel_angular_velocity_rt)
            
            # define the current configuration
            current_x = self.robot_configuration.x
            current_y = self.robot_configuration.y
            current_theta = self.robot_configuration.theta  
            q0 = np.array([
                current_x, 
                current_y,
                current_theta,
                driving_velocity,
                steering_velocity
            ])

            # Get the crowd positions: 
            if hasattr(self.crowd_motion_prediction_stamped_rt, 'crowd_motion_prediction') and \
                hasattr(self.crowd_motion_prediction_stamped_rt.crowd_motion_prediction, 'motion_predictions'):
                crowd = self.crowd_motion_prediction_stamped_rt.crowd_motion_prediction.motion_predictions
                crowd_position_array = np.array([[human.position.x, human.position.y] for human in crowd])
            else:
                crowd_position_array = np.array([])
            
            # Get position constraint index:
            elapsed_time = time.time() - self.current_time
            position_constraint_index = self.switcher.get_position_constraint_index(elapsed_time, self.pred_motion)
            # Update reference trajectories for NMPC:
            q_ref = self.update_reference(current_x, current_y)
            u_ref = np.zeros((self.nmpc_controller.nu, self.nmpc_controller.N))
            
            # Compute position error:
            position_error_x = current_x - q_ref[self.nmpc_controller.x_idx, 0]
            position_error_y = current_y - q_ref[self.nmpc_controller.y_idx, 0]
            pos_error = np.array([position_error_x, position_error_y])
            
            # Store the robot configuration, the position error, the crowd positions and the position constraint index:
            self.actual_config[self.config_idx] = q0
            self.position_error[self.config_idx] = pos_error
            self.crowd_position[self.config_idx] = crowd_position_array if crowd_position_array.size > 0 else None
            self.switch_index[self.config_idx] = position_constraint_index[0]
            # Run NMPC:
            self.nmpc_controller.update(q0, q_ref, u_ref, position_constraint_index, crowd)

            # Extract angular velocities of the wheels from NMPC control input:
            nmpc_control_input = self.nmpc_controller.get_command()
            alpha_r = nmpc_control_input[self.nmpc_controller.r_wheel_idx]
            alpha_l = nmpc_control_input[self.nmpc_controller.l_wheel_idx]

            # Compute driving and steering accelerations given angular velocities of the wheels:
            a_v = self.nmpc_controller.wheel_radius * 0.5 * (alpha_r + alpha_l)
            a_omega = (self.nmpc_controller.wheel_radius / self.nmpc_controller.wheel_separation) * (alpha_r - alpha_l)

            # Fill unicycle control input by integrating NMPC control inputs:
            self.unicycle_control_input.driving_velocity = driving_velocity + a_v * self.dt
            self.unicycle_control_input.steering_velocity = steering_velocity + a_omega * self.dt
            
            # Store the driving and steering accelerations, the NMPC control inputs and the update time:
            self.driving_accelerations[self.config_idx] = a_v
            self.steering_accelerations[self.config_idx] = a_omega
            self.control_inputs_nmpc[self.config_idx] = nmpc_control_input
            end_time = time.time()
            self.update_time[self.config_idx] = end_time - start_time
            self.config_idx += 1

            # check if the robot is arrived at the destination and display the results
            distance= np.sqrt(np.sum((self.waypoints[-1]-np.array([self.robot_configuration.x,
                                                                    self.robot_configuration.y]))**2))
            if distance < self.threshold:
                    self.save_simulation_data()
                    rospy.signal_shutdown("Reached the goal!")
            self.counter = 4
            self.failure_counter=0
            
            # Get the predicted motion from the NMPC controller
            self.pred_motion=self.nmpc_controller.get_predicted_motion()  
                
        except Exception as e:
            # If an exception is raised, stop the robot and store the driving and steering accelerations, the NMPC control inputs and the update time
            print("Exception: ", e) 
            self.failure_mem[self.config_idx]=0.05
            self.driving_accelerations[self.config_idx] = 0
            self.steering_accelerations[self.config_idx] = 0
            self.control_inputs_nmpc[self.config_idx] = np.zeros(self.nmpc_controller.nu)
            self.unicycle_control_input.driving_velocity = 0
            self.unicycle_control_input.steering_velocity = 0
            end_time = time.time()
            self.update_time[self.config_idx] = end_time - start_time
            self.config_idx += 1
            self.failure_counter+=1
            rospy.logwarn("Solver failure or other issue encountered, stopping the robot.")
            # If the robot is stuck for more than 200 iterations save the simulation data and stop the execution
            if self.failure_counter>200:
                self.save_simulation_data()
                rospy.signal_shutdown("Robot is stuck, execution terminated.")
    
def main():
    rospy.init_node('NMPCMotionGenerationManager', log_level=rospy.INFO)
    world_name = rospy.get_param("/world_name")
    motion_generation_manager = NMPCMotionGenerationManager(world_name)
    motion_generation_manager.start()
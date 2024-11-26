import matplotlib.pyplot as plt
import matplotlib.transforms as tr
from matplotlib.animation import FuncAnimation
import os
import numpy as np
from matplotlib.patches import Polygon
import argparse
import json
from tqdm import tqdm

class NMPCVisualizer:
    def __init__(self, filename) -> None:
        self.filename = filename
        plt.style.use('seaborn-darkgrid')
        # create the 'plots' directory if it doesn't exist and save the plot
        script_dir = os.path.dirname(__file__)
        save_dir = f'{self.filename}'
        img_dir = os.path.join(script_dir, '..', '..', save_dir)
        self.img_dir = os.path.normpath(img_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        
    def _plot_convex_areas(self, ax, switch_area_constraint_index, frame=None):
        
        if self.world == 'office':
            colors = ['orange', 'orange', 'orange']
        elif self.world == 'double':
            colors = ['orange', 'orange', 'orange','orange','orange','orange','orange','orange',]  # Example colors for different areas
        elif self.world == 'corridor':
            colors = ['orange', 'orange', 'orange', 'orange']
        alpha = 0.01  # Default semi-transparent alpha value
        # Clear previous areas to avoid overlaying them in each frame
        for patch in reversed(ax.patches):
            patch.remove()
        
        # Dynamically adjust transparency
        for i, vertices in enumerate(self.areas_vertices):
            # Check if the current area is the active constraint
            alpha = 0.3 if i == switch_area_constraint_index[frame] else 0.1
            polygon = Polygon(vertices, closed=True, color='blue', alpha=alpha, label=f'Area {i}')
            ax.add_patch(polygon)
            
            # Create and add the polygon for the current area
            polygon = Polygon(vertices, closed=True, color=colors[i], alpha=alpha, label=f'Area {i}')
            ax.add_patch(polygon)

    def _plot_wall(self, 
                   ax):
        # Define and plot the walls
        if self.world == 'office':
            walls = [((-5, -5), (-5, 10)),  # Left vertical wall
                    ((-5, 10), (5, 10)),   # Top horizontal wall
                    ((5, 10), (5, -5)),    # Right vertical wall
                    ((5, -5), (-5, -5)),   # Bottom horizontal wall
                    ((-5, 5), (1,5)),
                    ((2,5), (5,5))]
        elif self.world == 'double':
            walls = [((-5, -5), (-5, 10)),  
                    ((-5, -5), (5, -5)),
                    ((-5, 10), (5, 10)),  
                    ((5, 10), (5, -5)), 
                    ((5, 0), (3, 0)),
                    ((2, 0), (-5, 0)),
                    ((5, 5), (-2, 5)),
                    ((-5, 5), (-3, 5))]
        elif self.world == 'corridor':
            walls = [
                    ((-10, -5), (0, -5)),  # Left vertical wall
                    ((-10, -5), (-10, 0)), # Top left horizontal wall
                    ((0, -5), (0, 0)),      # Bottom left horizontal wall
                    ((0, 0), (-6, 0)),     # bottom left vertical
                    ((-10, 0), (-9, 0)),   # top left vertical
                    ((-9, 0), (-9, 5)),    # top center horizontal
                    ((-5, 0), (-5, 5)),    # bottom center hotizontal
                    ((-8, 5), (-10, 5)),   # top right vertical
                    ((-5, 5), (0, 5)),     # bottom right vertical
                    ((0, 5), (0, 10)),     # bottom right horizontal
                    ((-10, 5), (-10, 10)), # top right horizontal
                    ((0, 10), (-10, 10)),  # right vertical
            ]
        for wall_start, wall_end in walls:
            ax.plot([wall_start[0], wall_end[0]], 
                     [wall_start[1], wall_end[1]], 
                     'k-', 
                     linewidth=2)
            
    def _set_axis_properties(self, 
                             ax,  
                             xlabel, 
                             ylabel, 
                             title: str = None,
                             path: bool = False): 
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if path:
            if self.world == 'office':
                ax.set_xlim(-5,5)
                ax.set_ylim(-5,10)
                ax.set_aspect('equal', adjustable='box')
            elif self.world == 'double':
                ax.set_xlim(-5,5)
                ax.set_ylim(-5,10)
                ax.set_aspect('equal', adjustable='box')
            elif self.world == 'corridor':
                ax.set_xlim(-10,0)
                ax.set_ylim(-5,10)
                ax.set_aspect('equal', adjustable='box')
        if title is not None:
            ax.set_title(title)
        ax.grid(True)
        ax.legend()

    def create_animated_path_plot(self, waypoints, path, actual_paths, orientation, crowd_data, switch_area_constraint_index, interval=1, save=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_wall(ax)

        self._set_axis_properties(ax, title=f'Animated Robot Path - Heuristic: "{self.heuristic}"', xlabel='X Coordinate', ylabel='Y Coordinate', path=True)

        waypoints_x, waypoints_y = waypoints[:, 0], waypoints[:, 1]
        path_x, path_y = path[:, 0], path[:, 1]
        ax.plot(waypoints_x, waypoints_y, 'ro', label='Waypoints')
        ax.plot(path_x, path_y, 'b', label='Interpolated Path', linewidth=0.5, alpha=0.5)

        actual_path_line, = ax.plot([], [], 'g--', label='Actual Robot Path')
        robot_pos, = ax.plot([], [], 'wo', label='Robot_pos')
        orientation_tick, = ax.plot([], [], 'k-', linewidth=2) 

        if np.any(crowd_data):  
            crowd_positions = [ax.plot([], [], 'mo', alpha=0.5)[0] for _ in range(crowd_data.shape[1])]
        else:
            crowd_positions = []

        skip = 2

        def init():
            actual_path_line.set_data([], [])
            robot_pos.set_data([], [])
            orientation_tick.set_data([], []) 
            for crowd_pos in crowd_positions:
                crowd_pos.set_data([], [])
            return [actual_path_line, orientation_tick] + crowd_positions 

        def update(frame):
            frame_idx = frame * skip
            if frame_idx < len(actual_paths):
                actual_path_line.set_data(actual_paths[:frame_idx, 0], actual_paths[:frame_idx, 1])
            else:
                actual_path_line.set_data(actual_paths[:, 0], actual_paths[:, 1])
            robot_pos.set_data(actual_paths[frame_idx, 0], actual_paths[frame_idx, 1])
            robot_pos.set_markersize(26/2.54)
            if frame_idx < len(orientation):
                orientation_angle = orientation[frame_idx]
                end_x = actual_paths[frame_idx, 0] + np.cos(orientation_angle) * 0.3  
                end_y = actual_paths[frame_idx, 1] + np.sin(orientation_angle) * 0.3  
                orientation_tick.set_data([actual_paths[frame_idx, 0], end_x], [actual_paths[frame_idx, 1], end_y])
            if np.any(crowd_data) and frame_idx < crowd_data.shape[0]:
                for i, crowd_pos in enumerate(crowd_positions):
                    crowd_pos.set_data(crowd_data[frame_idx, i, 0], crowd_data[frame_idx, i, 1])
                    crowd_pos.set_markersize(40/2.54)
            self._plot_convex_areas(ax, switch_area_constraint_index, frame=frame_idx)
            return [actual_path_line, orientation_tick] + crowd_positions 
        anim = FuncAnimation(fig, update, frames=np.arange(1, (len(actual_paths)+skip-1)//skip), init_func=init, blit=False, interval=interval)
        if save:
            file_name = os.path.join(self.img_dir, 'robot_areaswitch_animation.mp4')
            anim.save(file_name, writer='ffmpeg', dpi=80, fps=30)
        else:
            plt.show()
    
    def create_robot_velocities_plot(self,
                                    driving_vel,
                                    steering_vel,
                                    save: bool = False):
        robot_velocities_fig, robot_velocities_axs = plt.subplots(figsize=(10, 6))
        robot_velocities_axs.plot(self.time,driving_vel, 'r-', label='Driving Velocity')
        self._set_axis_properties(robot_velocities_axs,
                                  xlabel='Time (s)',
                                  ylabel='Velocities')
        robot_velocities_axs.plot(self.time,steering_vel, 'b-', label='Steering Velocity')
        robot_velocities_axs.legend(('Driving Velocity (m/s)', 'Steering Velocity (rad/s)'))
        if save:
            self.save_plot(robot_velocities_fig, 'robot_velocities_subplot.png')

    def create_tracking_error_plot(self, 
                                   tracking_errors_x, 
                                   tracking_errors_y, 
                                   save: bool = False):
        tracking_error_fig, tracking_error_axs = plt.subplots(figsize=(10, 6))
        # Plot tracking error
        tracking_error_axs.plot(self.time,np.sqrt(tracking_errors_x**2+tracking_errors_y**2), 'b-', label='Tracking Error')
        self._set_axis_properties(tracking_error_axs,
                                  title=f'Tracking Error Over time - Heuristic: "{self.heuristic}"',
                                  xlabel='Time (s)',
                                  ylabel='Tracking Error (m)')
        if save:
            self.save_plot(tracking_error_fig, 'tracking_errors_subplot.png')

    def create_execution_time_plot(self, 
                                   execution_times, 
                                   failure_mem,
                                   save: bool=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        skip = 2
        iterations = np.arange(len(execution_times))
        ax.plot(iterations[::skip],failure_mem[::skip],  'r', label='NMPC Failures')
        ax.plot(iterations[::skip], execution_times[::skip],  'b', label='NMPC Execution Time')
        
        ax.set_title('NMPC Execution Time per Iteration')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Execution Time (s)')
        ax.grid(True)
        ax.legend()
        if save:
            self.save_plot(fig, 'nmpc_execution_times.png')

    def create_wheel_velocities_plot(self, 
                                     alpha_r, 
                                     alpha_l, 
                                     save: bool = False):
        input_plot_fig, input_plot_axs = plt.subplots(2, 1, figsize=(10, 12))
        skip = 2
        input_plot_axs[0].plot(self.time[::skip], alpha_l[::skip], 'r-', label='Left Wheel Angular Velocity')
        self._set_axis_properties(input_plot_axs[0],
                                  title=f'Left Wheel Angular Velocity - Heuristic: "{self.heuristic}"',
                                  xlabel='Time (s)',
                                  ylabel='Left Wheel Angular Velocity (rad/s)')

        input_plot_axs[1].plot(self.time[::skip], alpha_r[::skip], 'b-', label='Right Wheel Angular Velocity')
        self._set_axis_properties(input_plot_axs[1],
                                  title=f'Right Wheel Angular Velocity - Heuristic: "{self.heuristic}"',
                                  xlabel='Time (s)',
                                  ylabel='Right Wheel Angular Velocity (rad/s)')
        if save:
            self.save_plot(input_plot_fig, 'wheel_angular_velocities.png')

    def create_acceleration_plot(self,
                                 driving_acceleration,
                                 steering_acceleration,
                                 save: bool = False):
        acceleration_plot_fig, acceleration_plot_axs = plt.subplots(figsize=(10, 6))
        acceleration_plot_axs.plot(self.time, driving_acceleration, 'r-', label='Driving Acceleration')
        acceleration_plot_axs.plot(self.time, steering_acceleration, 'b-', label='Steering Acceleration')
        self._set_axis_properties(acceleration_plot_axs,
                                  xlabel='Time (s)',
                                  ylabel='Acceleration')
        acceleration_plot_axs.legend(('Driving Acceleration (m/s^2)', 'Steering Acceleration (rad/s^2)'))
        if save:
            self.save_plot(acceleration_plot_fig, 'acceleration_subplot.png')

    def save_plot(self, 
                  fig,
                  file_name):
        file_path = os.path.join(self.img_dir, file_name)
        fig.savefig(file_path)

    def show_plots(self):
        plt.show()
    
    def load_results(self):
        print("Loading Folder: ", self.filename)
        simulation = self.filename.split('/')[-1]
        path = os.path.join(self.img_dir, simulation)
        npz_filename = f"{path}.npz"
        json_filename = f"{path}_other_data.json"
        # Load the .npz file
        npz_data = np.load(npz_filename, allow_pickle=True)
        self.actual_config = npz_data['actual_config']
        self.crowd_position = npz_data['crowd_position']
        self.position_error = npz_data['position_error']
        self.switch_index = npz_data['switch_index']
        self.control_inputs_nmpc = npz_data['control_inputs_nmpc']
        self.driving_accelerations = npz_data['driving_accelerations']
        self.steering_accelerations = npz_data['steering_accelerations']
        self.update_time = npz_data['update_time']
        self.path = npz_data['path']
        self.failure_mem=npz_data['failure_mem']
        
        # Load the .json file
        with open(json_filename, 'r') as json_file:
            other_data = json.load(json_file)
        self.waypoints = np.array(other_data['waypoints'])
        self.heuristic = other_data['heuristic']
        self.world = other_data['world']
        self.num_people = other_data['num_people']
        self.dt=other_data['dt']
        self.areas_vertices=other_data['areas']
        # Handle other fields as needed
        print("World: ", self.world, "\nNumber of People: ", self.num_people, "\nHeuristic: ", self.heuristic)

        print(f"Loaded simulation data from {npz_filename} and {json_filename}")

    def display_final_results(self):
        actual_config = self.actual_config
        self.time=np.array([i * self.dt for i in range(len(actual_config))])

        # Represent progress with tqdm bar
        p_bar = tqdm(range(6), desc="Creating Plots", position=0, leave=True)

        # path, crowd and area animated plot 
        self.create_animated_path_plot(self.waypoints, 
                                       self.path, 
                                       self.actual_config[:, :2],
                                       self.actual_config[:, 2],
                                       self.crowd_position,
                                       self.switch_index,
                                       save=True)
        p_bar.update(1)
        
        # plot robot velocities
        self.create_robot_velocities_plot(self.actual_config[:, 3],
                                          self.actual_config[:, 4],
                                          save=True)
        p_bar.update(1)
        # tracking error plot
        self.create_tracking_error_plot(self.position_error[:, 0],
                                        self.position_error[:, 1],
                                        save=True)   
        p_bar.update(1)

        # control inputs plot
        self.create_wheel_velocities_plot(self.control_inputs_nmpc[:, 0],
                                          self.control_inputs_nmpc[:, 1],
                                          save=True)
        p_bar.update(1)
        # driving and steering accelerations plot
        self.create_acceleration_plot(self.driving_accelerations,
                                      self.steering_accelerations,
                                      save=True)
        p_bar.update(1)
        # nmpc execution time plot
        self.create_execution_time_plot(self.update_time,
                                        self.failure_mem,
                                        save=True)   
        p_bar.update(1)
        p_bar.close()

        self.show_plots()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NMPC results")
    parser.add_argument('-l', '--load', default="plots_v_cost/office_10_prediction", help="Load the results from the specified file")
    args = parser.parse_args()

    visualizer = NMPCVisualizer(args.load)
    visualizer.load_results()
    visualizer.display_final_results()





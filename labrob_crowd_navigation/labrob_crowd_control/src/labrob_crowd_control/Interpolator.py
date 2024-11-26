import numpy as np


class Interpolator:
    def __init__(self, 
                 world_name, 
                 avg_velocity, 
                 dt):
        self.world_name = world_name
        self.avg_velocity = avg_velocity
        self.dt = dt
        self.define_path()

    def define_path(self):
        # array of waypoints - defined by looking at simple office map (pal robotics)
        if self.world_name == 'office':
            self.waypoints = np.array([[-1.0, -2.0],
                                       [1.0, 0.0],
                                       [1.5, 2.5], 
                                       [1.5, 4.5],  
                                       [1.5, 7.0], 
                                       [-3.0, 8.0]])
        elif self.world_name == 'double':
            self.waypoints = np.array([[-3.0, -3.0],
                                       [-1.0, -3.0],
                                       [2.0, -3.5],
                                       [2.5, -2.5],
                                       [2.5, -1.0],
                                       [2.5, 2.0],
                                       [-2.5, 3.0],  
                                       [-2.5, 6.5],  
                                       [3.0, 8.0]])
        elif self.world_name == 'corridor':
            self.waypoints = np.array([[-3.0, -3.0],
                                       [-5.0, -3.0],
                                       [-7.5, -0.5],
                                       [-6.5, 3.0], 
                                       [-6.0, 7.5], 
                                       [-3.5, 8.0], 
                                       [-1.0, 8.0]])
        else:
            self.waypoints = np.array([[-1.0, -2.0],
                                       [1.0, 0.0],
                                       [1.5, 3.5], 
                                       [1.5, 4.5],  
                                       [1.5, 7.0], 
                                       [-3.0, 8.0]])
        self.interpolate_path()

    def interpolate_path(self):
        # compute the distance between waypoints
        diffs = np.diff(self.waypoints, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        # compute the time to travel between waypoints at the average velocity
        self.times = distances / self.avg_velocity
        # compute the number of points to use for each segment
        num_points_per_segment = np.ceil(self.times / self.dt).astype(int)
        # create a path by interpolating between waypoints
        path = [np.linspace(self.waypoints[i], self.waypoints[i+1], num=num_points, endpoint=False) 
                for i, num_points in enumerate(num_points_per_segment)]
        # concatenate the path segments
        path = np.concatenate(path)
        self.path = np.vstack([path, self.waypoints[-1]])
        
    def get_path(self):
        return self.path

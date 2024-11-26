import numpy as np
import time

from shapely.geometry import Point, Polygon

class AreaSwitcher:
    def __init__(self, 
                 nmpc_controller,
                 waypoints, 
                 heuristic: str = 'prediction',
                 world_name: str = 'office'):
        
        self.nmpc_controller = nmpc_controller
        self.heuristic = heuristic
        self.waypoints = waypoints
        self.world_name = world_name
        self.position_constraint_index = np.zeros(self.nmpc_controller.N, dtype=int)
        self.define_areas()
        self.strategy = self.select_strategy()

    def define_areas(self):
        if self.world_name == 'office':
            self.area0 = Polygon([(-4.7, -4.7), (4.7, -4.7), (4.7, 4.7), (-4.7, 4.7), (-4.7, -4.7)]) 
            self.area1 = Polygon([(1, -4), (2, -4), (2, 9), (1, 9), (1, -4)])
            self.area2 = Polygon([(-4.7, 5.3), (4.7, 5.3), (4.7, 9.7), (-4.7, 9.7), (-4.7, 5.3)]) 
            self.areas = [self.area0, self.area1, self.area2]
        if self.world_name == 'double':
            self.area0 = Polygon([(0, -3.5), (0, -0.3), (-4.7, -0.3), (-4.7, -4.7), (-3,-4.7),(0, -3.5)])
            self.area1 = Polygon([(4.7, -4.7), (4.7, -2.5), (-3.0, -2.5), (0.0, -4.7),(4.7, -4.7)])
            self.area2 = Polygon([(4.7, -4.7), (4.7, -0.3), (1.5, -0.3), (2.5, -4.7),(4.7, -4.7)])
            self.area3 = Polygon([(3, -2.0), (3, 4.7), (2, 4.7), (2, -2.0),(3, -2.0)])
            self.area4 = Polygon([(4.7, 0.5), (4.7, 3.0), (-1, 4.7), (-4.7, 4.7), (-4.7, 0.5), (4.7, 0.5)])
            self.area5 = Polygon([(-2.0, 0.5), (-2.0, 9.7), (-3.0, 9.7), (-3.0, 0.5),(-2, 0.5)])
            self.area6 = Polygon([(4.7, 5.5), (4.7, 9.5), (-0.5, 9.5), (-4.7, 6.5), (-4.7, 5.5),(4.7, 5.5)])
            self.areas = [self.area0, self.area1, self.area2, self.area3, self.area4, self.area5, self.area6]
        if self.world_name == 'corridor':  
            self.area0 = Polygon([(-2.0, 0.0), (-9.0, 0.0), (-9.0, -2.5), (-4.0, -4.7), (-2.0, -4.7), (-2.0, 0.0)])
            self.area1 = Polygon([(-8.7, -4.0), (-6.2, -4.0), (-6.2, 4.7), (-7.7, 4.7), (-8.7, 3.7), (-8.7, -4.0)])
            self.area2 = Polygon([(-7.7, 0.2), (-5.2, 0.2), (-5.2, 9.7), (-6.2, 9.7), (-7.7, 6.0), (-7.7, 0.2)])
            self.area3 = Polygon([(-4.0, 5.2), (-0.2, 7.0), (-0.2, 8.2), (-3.0, 9.7), (-7.0, 9.7), (-9.7, 5.2), (-4.0, 5.2)])     
            self.areas = [self.area0, self.area1, self.area2, self.area3]   
        self.areas_num = len(self.areas)
    
    def select_strategy(self):

        if self.heuristic == 'prediction':
            return self.get_position_constraint_index_prediction
        elif self.heuristic == 'next_waypoint':
            return self.get_position_constraint_index_next
        elif self.heuristic == 'nearest_waypoint':
            return self.get_position_constraint_index_nearest
        elif self.heuristic == 'time':
            return self.get_position_constraint_index_time

    def get_position_constraint_index_prediction(self, elapsed_time, pred_motion):
        
        if pred_motion is None:
                return self.position_constraint_index
        for area_index, area in enumerate(self.areas):
            if all(area.contains(Point(point)) or area.boundary.contains(Point(point)) for point in pred_motion):
                self.position_constraint_index = [area_index] * len(pred_motion)
        return self.position_constraint_index
    
    def get_position_constraint_index_next(self, elapsed_time, pred_motion):
        if pred_motion is None:
            return self.position_constraint_index
        threshold = 0.2
        if self.world_name == 'office':
            switch_wp=np.array([self.waypoints[2],
                                self.waypoints[4]])
        elif self.world_name == 'double':
            switch_wp = np.array([self.waypoints[1],
                                  self.waypoints[3],
                                  self.waypoints[4],
                                  self.waypoints[5],
                                  self.waypoints[6], 
                                  self.waypoints[7]])
        elif self.world_name == 'corridor':
            switch_wp=np.array([self.waypoints[2],
                                self.waypoints[3],
                                self.waypoints[4]])
        for point_index, point in enumerate(pred_motion):
            for area_num in range(self.areas_num-1):
                if (self.position_constraint_index[point_index]==area_num and np.sum((switch_wp[area_num]-point[:2])**2)<threshold):
                    self.position_constraint_index = np.ones(self.nmpc_controller.N + 1,dtype=int)*area_num
                    self.position_constraint_index[point_index:]=area_num+1
        last_pos_index = self.position_constraint_index[-1]
        if self.position_constraint_index[0]==last_pos_index-1 and self.areas[last_pos_index].contains(Point(pred_motion[0])):
            self.position_constraint_index = np.ones(self.nmpc_controller.N + 1,dtype=int)*last_pos_index
        return self.position_constraint_index
            
    def get_position_constraint_index_nearest(self, elapsed_time, pred_motion):
        if pred_motion is None:
            return self.position_constraint_index
    
        for i, point in enumerate(pred_motion):
            point_t = point[:2]
            if self.world_name == 'double':
                if point[:2][0] == 0 and point[:2][1] == 0:
                    point_t = [-3, -3]
            distances = [np.linalg.norm(np.array(point_t) - np.array(waypoint)) for waypoint in self.waypoints]
            closest_waypoint_index = np.argmin(distances)
                
            if self.world_name == 'office':
                if closest_waypoint_index == 0 or closest_waypoint_index == 1 or closest_waypoint_index == 2:
                    self.position_constraint_index[i] = 0  
                elif closest_waypoint_index == 3:
                    self.position_constraint_index[i] = 1  
                else:
                    self.position_constraint_index[i] = 2  
            
            elif self.world_name == 'double':
                if closest_waypoint_index == 0:
                    self.position_constraint_index[i] = 0  
                elif closest_waypoint_index == 1 or closest_waypoint_index == 2:
                    self.position_constraint_index[i] = 1 
                elif closest_waypoint_index == 3:
                    self.position_constraint_index[i] = 2
                elif closest_waypoint_index == 4:
                    self.position_constraint_index[i] = 3
                elif closest_waypoint_index == 5:
                    self.position_constraint_index[i] = 4
                elif closest_waypoint_index == 6:
                    self.position_constraint_index[i] = 5  
                else:
                    self.position_constraint_index[i] = 6

            elif self.world_name == 'corridor':
                if closest_waypoint_index == 0 or closest_waypoint_index == 1:
                    self.position_constraint_index[i] = 0  
                elif closest_waypoint_index == 2:
                    self.position_constraint_index[i] = 1  
                elif closest_waypoint_index == 3:
                    self.position_constraint_index[i] = 2  
                else:
                    self.position_constraint_index[i] = 3
        last_pos_index = self.position_constraint_index[-1]
        if self.position_constraint_index[0]==last_pos_index-1 and self.areas[last_pos_index].contains(Point(pred_motion[0])):
            self.position_constraint_index = np.ones(self.nmpc_controller.N + 1,dtype=int)*last_pos_index
        # sometimes the last index is not updated correctly
        if self.position_constraint_index[-1]!=self.position_constraint_index[-2]:
            self.position_constraint_index[-1]=self.position_constraint_index[-2]
        return self.position_constraint_index
        
    def get_position_constraint_index_time(self, elapsed_time, pred_motion):
        if pred_motion is None:
            return self.position_constraint_index
        if self.world_name == 'office':
            if elapsed_time < 20:
                area_index = 0
            elif elapsed_time < 25:
                area_index = 1
            else:
                area_index = 2
            self.position_constraint_index[:] = area_index
        elif self.world_name == 'double':
            if elapsed_time < 8:
                area_index = 0
            elif elapsed_time < 18:
                area_index = 1
            elif elapsed_time < 22:
                area_index = 2
            elif elapsed_time < 28:
                area_index = 3
            elif elapsed_time < 40:
                area_index = 4
            elif elapsed_time < 50:
                area_index = 5
            else:
                area_index = 6
            self.position_constraint_index[:] = area_index
        elif self.world_name == 'corridor':
            if elapsed_time < 12:
                area_index = 0
            elif elapsed_time < 20:
                area_index = 1
            elif elapsed_time < 36:
                area_index = 2
            else:
                area_index = 3
            self.position_constraint_index[:] = area_index
        return self.position_constraint_index

    def get_position_constraint_index(self,
                                      elapsed_time,
                                      pred_motion=None):  
        return self.strategy(elapsed_time, pred_motion)
            


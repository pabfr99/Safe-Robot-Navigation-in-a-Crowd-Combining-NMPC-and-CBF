import math
import threading
import numpy as np
import rospy
np.random.seed(42)

import gazebo_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg

import labrob_crowd_navigation_msgs.msg
import labrob_crowd_navigation_utils.utils

from labrob_crowd_perception.FASMachine import *
import labrob_crowd_perception.CommonVars as CommonVars
if CommonVars.TRAJECTORY_SAVE:
    from scipy.io import savemat
from scipy.spatial.distance import cdist
from sklearn.cluster import MeanShift,KMeans, estimate_bandwidth, DBSCAN

def cluster_scans(scans):
    '''
    Simple method to cluster the scans. In this way close scan are grouped toghether;
    Input: array of scans:
    Output:
    Array of clusters
    '''
    clusters = []
    cluster = []
    for (index,distance) in scans:
        if distance == np.inf:
            if cluster:
                cluster.sort(key = lambda x: x[1], reverse = False)
                clusters.append((cluster[0][0], cluster[0][1]))
                cluster = []
        else:
            cluster.append((index, distance))
    
    if CommonVars.CROWD_PRINT:
        print(clusters)
    if len(clusters) > 0:
        clusters.sort(key = lambda x: x[1], reverse = False)
        clusters = clusters[0:CommonVars.K_CLUSTERS]
    #print("clusters:", clusters)
    return clusters

def cluster_scans_k_means(scans,tiago_conf,angmin, angincr):
    scans_xy_abs = []
    scans_polar = []
    cluster_number = CommonVars.K_CLUSTERS
    for element in scans:
        if element[1] != np.inf:
            scan_idx = element[0]
            scan_dist = element[1]
            angle_laserscan = angmin + (scan_idx * angincr) 
            measure = np.array([scan_dist*np.cos(angle_laserscan+tiago_conf.theta) + tiago_conf.x ,\
                            scan_dist*np.sin(angle_laserscan+tiago_conf.theta) +  tiago_conf.y])
            scans_xy_abs.append(measure)
            scans_polar.append(element)


    if len(scans_xy_abs) != 0:
        #k_means = MeanShift(bandwidth = 1)

        dynamic_cluster_number = min(CommonVars.NUM_ACTORS, len(scans_xy_abs))
        #k_means = KMeans(n_clusters=dynamic_cluster_number)

        #bw = estimate_bandwidth(np.array(scans_xy_abs), quantile=0.5, n_samples=None, random_state=0, n_jobs=None)
        #print("BANDWIDTH: ", bw)
        #k_means = MeanShift(bandwidth = 1)
        k_means = DBSCAN(eps=1, min_samples=2)
        scan_to_cluster = k_means.fit_predict(np.array(scans_xy_abs))

        dynamic_cluster_number =max(scan_to_cluster)+1 # SHOULD WORK FOR DBSCAN AND MEANSHIFT
        if(min(scan_to_cluster) == -1):
            print("PROBLEM IT SAYS NOISY ")
            #exit()
    
       
        if CommonVars.CROWD_PRINT:
            print(scan_to_cluster)
            print(dynamic_cluster_number)

        cluster_collection = np.zeros((dynamic_cluster_number,2))
        cluster_polar_collection = np.zeros((dynamic_cluster_number,2))
        
        for cluster_num in range(dynamic_cluster_number):
            minimum_distance = 999
            for id,(index_scan,cluster_polar) in zip(scan_to_cluster,enumerate(scans_polar)):
                if id == cluster_num:
                    if cluster_polar[1] < minimum_distance:
                        minimum_distance = cluster_polar[1]
                        cluster_collection[cluster_num] = scans_xy_abs[index_scan]
                        cluster_polar_collection[cluster_num] = scans_polar[index_scan]
        
        #cluster_polar_collection = merge_clusters(cluster_polar_collection)

        cluster_polar_collection = cluster_polar_collection.tolist()
        cluster_polar_collection.sort( key = lambda x: x[1],reverse = False)
        cluster_polar_collection = cluster_polar_collection[0:CommonVars.K_CLUSTERS]
    else:
        cluster_polar_collection = np.array([])

    return cluster_polar_collection


def cluster_to_xy_abs(cluster,tiago,angmin,anginc):
    scan_idx = cluster[0]
    scan_dist = cluster[1]
    angle_laserscan = angmin  + (scan_idx * anginc) 
    x_cl = scan_dist*np.cos(angle_laserscan+tiago.theta) + tiago.x 
    y_cl = scan_dist*np.sin(angle_laserscan+tiago.theta) +  tiago.y
    return np.array([x_cl,y_cl])
                       
# def merge_clusters(polar_clusters):
#     # Create an empty list to hold the merged clusters
#     temp_polar_cluster_list = np.copy(polar_clusters).tolist()
#     merged_polar_clusters = []
#     print("unmerged clusters:",polar_clusters )

#     # Loop through each pair of clusters
#     while(len(temp_polar_cluster_list) > 0):
#         #extract first cluster available
#         cluster = temp_polar_cluster_list.pop(0)
#         matched_cluster = cluster
#         if(len(temp_polar_cluster_list) > 0):
#             for j,other_cluster in enumerate(temp_polar_cluster_list):
#                 #test if abs value of scan id is under a certain value
#                 iddiff = abs(cluster[0]-other_cluster[0]) 
#                 #test if abs value of polar distance from tiago is under a certain value
#                 rhodiff = abs(cluster[1]-other_cluster[1])

#                 if(iddiff < CommonVars.ID_DIFF_ABSVAL):
#                     if(rhodiff< CommonVars.RHO_DIFF_ABSVAL):
#                         print("cluster",cluster, "has distances id:",iddiff,"rho:",rhodiff,"with cluster:",other_cluster)
#                         #lets take the closest possible cluster between the two
#                         if(cluster[1] < other_cluster[1]):
#                             matched_cluster = cluster
#                         else:
#                             matched_cluster = other_cluster
#                         _ = temp_polar_cluster_list.pop(j)
#                         break
#         merged_polar_clusters.append(matched_cluster)

#     print("merged clusters:", merged_polar_clusters)
#     return merged_polar_clusters


    # for i in range(len(temp_cluster_list)):
    #     for j in range(len(temp_cluster_list)):
    #         if (i != j):
    #         # Calculate the Euclidean distance between the two clusters
    #             distance = np.linalg.norm(temp_cluster_list[i] - temp_cluster_list[j])
    #             print("cluster",i,temp_cluster_list[i],temp_polar_cluster_list[i], "has distance: ",distance,"with cluster:",j,temp_cluster_list[j], temp_polar_cluster_list[j])
    #             # If the distance is less than the threshold, merge the clusters
    #             if distance < distance_threshold:
        
    #                 closest_cluster = i
    #                 if(polar_clusters[i][1] < polar_clusters[j][1]):
    #                     closest_cluster = i
    #                 else:
    #                     closest_cluster = j

    #                 merged_clusters.append(cluster_list[closest_cluster])
    #                 merged_polar_clusters.append(polar_clusters[closest_cluster])
    #                 # Remove the individual clusters from the cluster list
    #                 temp_cluster_list = np.delete(temp_cluster_list, [i,j], axis=0)
    #                 temp_polar_cluster_list = np.delete(temp_polar_cluster_list, [i,j], axis=0)

    #                 break

    # # Add any clusters that were not merged to the list of merged clusters
    # print("merged partial clusters:", merged_polar_clusters)
    # merged_clusters.extend(temp_cluster_list)
    # merged_polar_clusters.extend(temp_polar_cluster_list)
    # print("unmerged clusters:",polar_clusters )
    # print("merged clusters:", merged_polar_clusters)

    # return merged_polar_clusters

    

#Tiago position acquisition
class CrowdPerceptionLaserScan:
    '''
    Main class of this project. It is implemented as described in the paper.
    Actor names: name of the actors.
    robot_configuration_non_rt: Tiago configuration
    tiago_velocity: Tiago velocity
    actor_configurations: Actors' configurations
    laser_scan: Array of measurements from the laser scan
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        self.robot_configuration = labrob_crowd_navigation_utils.utils.Configuration(
            0.0, 0.0, 0.0
        )
        self.tiago_velocity = labrob_crowd_navigation_utils.utils.TiagoVelocity(0.0,0.0,0.0)
        self.actor_configurations = {}
        self.laser_scan = None
        self.K = CommonVars.K_CLUSTERS
        self.NActors = CommonVars.NUM_ACTORS
        self.previous_clusters = None
        self.previous_time = 0

        self.actor_names = ['actor_{}'.format(i) for i in range(self.NActors)]

        self.all_actor_configurations = {key: list() for key in self.actor_names}
        self.all_tiago_positions = []
        self.kalman_infos = {}
        kalman_names = ['KF_{}'.format(i) for i in range(self.K)]
        self.kalman_infos = {key: list() for key in kalman_names}

        # Subscribers:
        rospy.Subscriber(
            '/gazebo/model_states',
            gazebo_msgs.msg.ModelStates,
            self.gazebo_model_states_callback
        )
        rospy.Subscriber(
            '/scan_raw',
            sensor_msgs.msg.LaserScan,
            self.laser_scan_callback
        )
        rospy.Subscriber(
            '/mobile_base_controller/cmd_vel',
            geometry_msgs.msg.Twist,
            self.unycicle_input_callback
        )


    def log_values(self):
        output_dict = {}
        output_dict['tiago'] = np.array(self.all_tiago_positions)
        output_dict['humans'] = self.all_actor_configurations
        output_dict['kfs'] = self.kalman_infos
        savemat('/tmp/log_pos_cp.mat', output_dict)

    def unycicle_input_callback(self, geometry_msg):
        linear_velocity = geometry_msg.linear
        angular_velocity = geometry_msg.angular
        tiago_velocity = labrob_crowd_navigation_utils.utils.TiagoVelocity(
            linear_velocity.x,
            linear_velocity.y,
            angular_velocity.z
        )
        self.data_lock.acquire()
        self.tiago_velocity = tiago_velocity
        self.data_lock.release()

    def gazebo_model_states_callback(self, gazebo_model_states_msg):
        actor_configurations = {}
        #Tiago
        if 'tiago' in gazebo_model_states_msg.name:
            husky_idx = gazebo_model_states_msg.name.index('tiago')
            p = gazebo_model_states_msg.pose[husky_idx].position
            q = gazebo_model_states_msg.pose[husky_idx].orientation
            v = gazebo_model_states_msg.twist[husky_idx].linear
            robot_configuration_nonrt =  labrob_crowd_navigation_utils.utils.Configuration(
                p.x,
                p.y,
                math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                )
            )
        for actor_name in self.actor_names:
            if actor_name in gazebo_model_states_msg.name:
                actor_idx = gazebo_model_states_msg.name.index(actor_name)
                p = gazebo_model_states_msg.pose[actor_idx].position
                q = gazebo_model_states_msg.pose[actor_idx].orientation
                actor_configuration = labrob_crowd_navigation_utils.utils.Configuration(
                    p.x,
                    p.y,
                    math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    )
                )            
                actor_configurations[actor_name] = actor_configuration

        self.data_lock.acquire()
        self.actor_configurations = actor_configurations
        self.robot_configuration = robot_configuration_nonrt
        self.data_lock.release()

    def laser_scan_callback(self, laser_scan_msg):
        self.data_lock.acquire()
        self.laser_scan = \
            labrob_crowd_navigation_utils.utils.LaserScan.from_message(laser_scan_msg)
        self.data_lock.release()

    def start(self):
        rate = rospy.Rate(50) # 100 Hz
        self.previous_time =  rospy.get_time()
        if CommonVars.TRAJECTORY_SAVE:
            rospy.on_shutdown(self.log_values)

        # Setting up publishers:
        crowd_motion_prediction_publisher = rospy.Publisher(
            'crowd_motion_prediction',
            labrob_crowd_navigation_msgs.msg.CrowdMotionPredictionStamped,
            queue_size=1
        )
        #kalman_init = False

        fsms = [FASMachine() for i in range(self.K)]
        while not rospy.is_shutdown():

            # Current time:
            time = rospy.get_time()

            if(self.actor_configurations == {} or self.laser_scan is None):
                print("NO INFOS YET")
                rate.sleep()
                continue

            deltat = time - self.previous_time
            self.previous_time =  time
            print("----------DT=",deltat)

            if CommonVars.TRAJECTORY_SAVE:
                for actor_name in self.actor_configurations.keys():
                    self.all_actor_configurations[actor_name].append(np.array([self.actor_configurations[actor_name].x, self.actor_configurations[actor_name].y, time]))
                self.all_tiago_positions.append(np.array([self.robot_configuration.x, self.robot_configuration.y, time]))

            if CommonVars.FAKE_FSM:

                crowd_motion_prediction = \
                    labrob_crowd_navigation_utils.utils.CrowdMotionPrediction()
                for actor_name, actor_configuration in self.actor_configurations.items():
                    
                    if CommonVars.CROWD_PRINT:   rospy.loginfo('{} at configuration {}'.format(actor_name, actor_configuration))
                    crowd_motion_prediction.append(
                    labrob_crowd_navigation_utils.utils.MotionPrediction(
                        labrob_crowd_navigation_utils.utils.Position(actor_configuration.x,actor_configuration.y,0),
                        labrob_crowd_navigation_utils.utils.Velocity(0,0,0)
                    )
                )
            else:
                #reset crowd_motion_prediction message
                crowd_motion_prediction = \
                    labrob_crowd_navigation_utils.utils.CrowdMotionPrediction()
                
                #We remove the first 20 items from start and end of tiago because it is miss-placed
                OFFSET = 20
                ANGMIN = self.laser_scan.angle_min
                ANGMAX = self.laser_scan.angle_max
                ANGINCR = self.laser_scan.angle_increment

                extended_scanlist = []
                for index, value in enumerate(self.laser_scan.ranges):
                    extended_scanlist.append((index,value))
                trimmed_scanlist = np.delete(extended_scanlist,range(OFFSET),0)
                trimmed_scanlist = np.delete(trimmed_scanlist, range( len(trimmed_scanlist) -OFFSET , len(trimmed_scanlist)),0)
                
                #create clusters ordered in descending distance order
                #laserscan = cluster_scans(trimmed_scanlist)
                laserscan = cluster_scans_k_means(trimmed_scanlist, self.robot_configuration,ANGMIN,ANGINCR)

                if CommonVars.CROWD_PRINT:
                    print("oderdered clusters", laserscan)
                dynamic_clusters = np.copy(laserscan)

                # Create array of predicted states for all FSMS
                for i,clus in enumerate(dynamic_clusters):
                    dynamic_clusters[i] = cluster_to_xy_abs(dynamic_clusters[i], self.robot_configuration, ANGMIN, ANGINCR)
                predicted_states = np.zeros((self.K,2))
                used_clusters = []

                for i,fsm in enumerate(fsms):
                    if fsm.next_time_pass_to in ('active', 'hold'):
                        future_state_arr = fsm.propagate_state(time, fsm.current_state, 1, deltat)[0]
                        predicted_states[i] = future_state_arr[0:2]
                    elif  fsm.next_time_pass_to == 'start':
                        future_state_arr = fsm.current_state
                        predicted_states[i] = future_state_arr[0:2]
                    else:
                        future_state_arr = CommonVars.KALMAN_NULLSTATE
                        predicted_states[i] = future_state_arr[0:2]

                # Compute pairwise distances between predicted states and dynamic clusters
                if(dynamic_clusters.shape[0] > 0):
                    distances = cdist(predicted_states, dynamic_clusters)
                else:
                    distances = None
                
                if CommonVars.CROWD_PRINT:
                    print("predstates:", predicted_states)
                    print("newclusters:",dynamic_clusters)
                    print("dist matrix: PREDxNEW", distances)

                # Match each FSM to the closest available cluster
                for i,fsm in enumerate(fsms):

                    state = fsm.next_time_pass_to

                    if CommonVars.TRAJECTORY_SAVE:
                        self.kalman_infos['KF_{}'.format(i)].append(np.array([fsm.current_state, state, time]))

                    if CommonVars.CROWD_PRINT:
                        print("FSM:",i, "is",state, "future:",predicted_states[i])  
                        print("unavailable measures:", used_clusters)

                    # Find the closest available cluster to the FSM's predicted state
                    min_dist = np.inf
                    min_row = None
                    if(type(distances) != type(None)):
                        for j in range(distances.shape[1]):
                            # Check if cluster is still available
                            if j in used_clusters:
                                continue
                            # Find distance between predicted state and cluster
                            dist = distances[i][j]
                            if dist < min_dist:
                                min_dist = dist
                                min_row = j
                    if min_row is not None:
                        # Match FSM to closest available cluster
                        cluster = dynamic_clusters[min_row]
                        measure = cluster#cluster_to_xy_abs(cluster, self.robot_configuration, ANGMIN, ANGINCR)
                        fsm.choose_state(time, measure, self.robot_configuration)
                        # Mark cluster as used
                        used_clusters.append(min_row)

                        if CommonVars.CROWD_PRINT:
                            print("matched to cluster: ",min_row, dynamic_clusters[min_row], "abs:",measure, "dist:",min_dist)
                    else:
                        # No available clusters, reset FSM
                        if CommonVars.CROWD_PRINT:
                            print("FSM:",i, "ran out of measurements! SAD")
                        measure = np.array([0,0])
                        fsm.choose_state(time, measure, self.robot_configuration)

                    current_state =  fsm.current_state
                    pos_i = labrob_crowd_navigation_utils.utils.Position(
                        current_state[0],
                        current_state[1],
                        0
                    )
                    vel_i = labrob_crowd_navigation_utils.utils.Velocity(current_state[2], current_state[3], i)

                    crowd_motion_prediction.append(
                        labrob_crowd_navigation_utils.utils.MotionPrediction(
                            pos_i,
                            vel_i
                        )
                    )

                    propagated_states = fsms[i].propagate_state(time,current_state,CommonVars.N_PREDICTIONS,CommonVars.N_PREDICTIONS_DELTA)

                    if CommonVars.CROWD_PRINT:
                        fsms[i].debug()
                        print(propagated_states)

                    for  j in range(0,CommonVars.N_PREDICTIONS):
                        current_state =  propagated_states[j]
                        pos_i = labrob_crowd_navigation_utils.utils.Position(
                        current_state[0],
                        current_state[1],
                        0
                        )
                        vel_i = labrob_crowd_navigation_utils.utils.Velocity(current_state[2], current_state[3], i*10 + j)

                        crowd_motion_prediction.append(
                            labrob_crowd_navigation_utils.utils.MotionPrediction(
                                pos_i,
                                vel_i
                            )
                        )


            crowd_motion_prediction_stamped = \
                labrob_crowd_navigation_utils.utils.CrowdMotionPredictionStamped(
                    rospy.Time.from_sec(time), 'map', crowd_motion_prediction
                )
            crowd_motion_prediction_stamped_msg = \
                labrob_crowd_navigation_utils.utils.CrowdMotionPredictionStamped.to_message(
                    crowd_motion_prediction_stamped
                )
            crowd_motion_prediction_publisher.publish(
                crowd_motion_prediction_stamped_msg
            )

            rate.sleep()

def main():
    rospy.init_node('CrowdPerceptionLaserScan', log_level=rospy.INFO)

    crowd_prediction_manager = CrowdPerceptionLaserScan()
    crowd_prediction_manager.start()

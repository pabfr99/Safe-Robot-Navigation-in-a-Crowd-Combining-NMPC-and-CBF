import math
import threading
import rospy

import gazebo_msgs.msg
import tf2_ros

import labrob_crowd_navigation_msgs.msg
import labrob_crowd_navigation_utils.utils

class CrowdPerceptionGroundTruth:

    def __init__(self):
        self.controller_frequency = 50 # [Hz]

        self.data_lock = threading.Lock()
        

        # Setting up reference frames:
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'

        # Setting up TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.robot_configuration = labrob_crowd_navigation_utils.utils.Configuration(
            -3.0, -3.0, 0.0
        )

        self.actor_configurations = {}
        self.previous_actor_configurations = {}

        # Subscribers:
        rospy.Subscriber(
            '/gazebo/model_states',
            gazebo_msgs.msg.ModelStates,
            self.gazebo_model_states_callback
        )

    def gazebo_model_states_callback(self, gazebo_model_states_msg):
        actor_configurations = {}
        for gazebo_model_name in gazebo_model_states_msg.name:
            if gazebo_model_name.startswith('actor_'):
                actor_idx = gazebo_model_states_msg.name.index(gazebo_model_name)
                p = gazebo_model_states_msg.pose[actor_idx].position
                q = gazebo_model_states_msg.pose[actor_idx].orientation
                actor_configuration = labrob_crowd_navigation_utils.utils.Configuration(
                    p.x,
                    p.y,
                    math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    )
                )            
                actor_configurations[gazebo_model_name] = actor_configuration
                
               

        self.data_lock.acquire()
        self.actor_configurations = actor_configurations
        self.data_lock.release()

    def calculate_velocity(self, current_config, actor_configuration, time_diff):

        if time_diff > 0 and len(self.previous_actor_configurations)>0:
            previous_config = self.previous_actor_configurations[actor_configuration]
            vx = (current_config.x - previous_config.x) * self.controller_frequency
            vy = (current_config.y - previous_config.y) * self.controller_frequency
            #print(vx, vy)
            return vx, vy
        return 0.0, 0.0

    
    def start(self):
        rate = rospy.Rate(self.controller_frequency)

        # Setting up publishers:
        crowd_motion_prediction_publisher = rospy.Publisher(
            'crowd_motion_prediction',
            labrob_crowd_navigation_msgs.msg.CrowdMotionPredictionStamped,
            queue_size=1
        )

        while not rospy.is_shutdown():

            # Current time:
            time = rospy.get_time()

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_footprint_frame, rospy.Time()
                )
                self.robot_configuration = \
                    labrob_crowd_navigation_utils.utils.Configuration.set_from_tf_transform(transform)
                
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                rate.sleep()
                continue

            if self.actor_configurations != {}:
                crowd_motion_prediction = \
                    labrob_crowd_navigation_utils.utils.CrowdMotionPrediction()
                for actor_configuration in self.actor_configurations.keys():
                    vx, vy = self.calculate_velocity(self.actor_configurations[actor_configuration],
                                                     actor_configuration,
                                                     1/self.controller_frequency)

                    crowd_motion_prediction.append(
                    labrob_crowd_navigation_utils.utils.MotionPrediction(
                        labrob_crowd_navigation_utils.utils.Position(
                            self.actor_configurations[actor_configuration].x,
                            self.actor_configurations[actor_configuration].y,
                            0.0
                        ),
                        labrob_crowd_navigation_utils.utils.Velocity(
                            vx,
                            vy,
                            0.0
                        )
                    )
                )
                self.previous_actor_configurations = self.actor_configurations

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
    rospy.init_node('CrowdPerceptionGroundTruth', log_level=rospy.INFO)

    crowd_prediction_manager = CrowdPerceptionGroundTruth()
    crowd_prediction_manager.start()

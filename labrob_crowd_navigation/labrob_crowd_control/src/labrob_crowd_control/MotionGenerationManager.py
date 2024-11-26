import rospy
import tf2_ros
import threading

import geometry_msgs.msg
import sensor_msgs.msg

import labrob_crowd_navigation_msgs.msg
import labrob_crowd_navigation_utils.utils


class MotionGenerationManager:
    def __init__(self, world_name):
        # Frequency of the control loop:
        self.controller_frequency = 20 # [Hz]
        self.dt = 1.0 / self.controller_frequency # [s]

        if world_name == 'office' or world_name == 'corridor':
            self.robot_configuration = labrob_crowd_navigation_utils.utils.Configuration(
                -3.0, -3.0, 0.0
            )
        elif world_name == 'double':
            self.robot_configuration = labrob_crowd_navigation_utils.utils.Configuration(
                4.0, -4.0, 0.0
            )
        else:
            self.robot_configuration = labrob_crowd_navigation_utils.utils.Configuration(
                0.0, 0.0, 0.0
            )

        # Control input set to zero:
        self.unicycle_control_input = labrob_crowd_navigation_utils.utils.UnicycleControlInput(
            0.0, 0.0
        )

        # Setting up reference frames:
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'

        # Setting up TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Data lock to safely manage callbacks:
        self.data_lock = threading.Lock()

        # Real-time and non real-time resources:
        self.crowd_motion_prediction_stamped_nonrt = labrob_crowd_navigation_utils.utils.CrowdMotionPredictionStamped(
                rospy.Time.now(),
                self.map_frame,
                labrob_crowd_navigation_utils.utils.CrowdMotionPrediction()
        )
        self.crowd_motion_prediction_stamped_rt = self.crowd_motion_prediction_stamped_nonrt

        self.left_wheel_angular_velocity_nonrt = 0.0 # [rad/s]
        self.left_wheel_angular_velocity_rt = self.left_wheel_angular_velocity_nonrt # [rad/s]
        self.right_wheel_angular_velocity_nonrt = 0.0 # [rad/s]
        self.right_wheel_angular_velocity_rt = self.right_wheel_angular_velocity_nonrt # [rad/s]

        # Subscribers:
        rospy.Subscriber(
            "crowd_motion_prediction",
            labrob_crowd_navigation_msgs.msg.CrowdMotionPredictionStamped,
            self.crowd_motion_prediction_stamped_callback
        )
        rospy.Subscriber(
            '/joint_states',
            sensor_msgs.msg.JointState,
            self.joint_states_callback
        )

    def crowd_motion_prediction_stamped_callback(self, crowd_motion_prediction_stamped_msg):
        crowd_motion_prediction_stamped_nonrt = \
            labrob_crowd_navigation_utils.utils.CrowdMotionPredictionStamped.from_message(
                crowd_motion_prediction_stamped_msg
            )
        self.data_lock.acquire()
        self.crowd_motion_prediction_stamped_nonrt = \
            crowd_motion_prediction_stamped_nonrt
        self.data_lock.release()
    
    def joint_states_callback(self, msg):
        wheel_left_joint_idx = msg.name.index('wheel_left_joint')
        wheel_right_joint_idx = msg.name.index('wheel_right_joint')
        self.data_lock.acquire()
        self.left_wheel_angular_velocity_nonrt = msg.velocity[wheel_left_joint_idx]
        self.right_wheel_angular_velocity_nonrt = msg.velocity[wheel_right_joint_idx]
        self.data_lock.release()

    def start(self):
        rate = rospy.Rate(self.controller_frequency)

        # Setting up publishers:
        cmd_vel_publisher = rospy.Publisher(
            '/mobile_base_controller/cmd_vel',
            geometry_msgs.msg.Twist,
            queue_size=1
        )
        crowd_prediction_publisher = rospy.Publisher(
            '/crowd_motion_prediction',
            labrob_crowd_navigation_msgs.msg.CrowdMotionPrediction,
            queue_size=1
        )

        while not rospy.is_shutdown():
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

            # Copy non real-time data to real-time data if possible:
            if self.data_lock.acquire(False):
                self.crowd_motion_prediction_stamped_rt = self.crowd_motion_prediction_stamped_nonrt
                self.left_wheel_angular_velocity_rt = self.left_wheel_angular_velocity_nonrt
                self.right_wheel_angular_velocity_rt = self.right_wheel_angular_velocity_nonrt
                self.data_lock.release()

            # Call update of child class, which fills driving and steering velocity:
            self.update()

            # Create a twist ROS message:
            cmd_vel_msg = labrob_crowd_navigation_utils.utils.UnicycleControlInput.to_twist_message(
                self.unicycle_control_input
            )
            # Create a crowd prediction ROS message:
            # uses the crowd_motion_prediction_stamped_rt, copy of real-time data
            crowd_prediction_msg = labrob_crowd_navigation_utils.utils.CrowdMotionPrediction.to_message(
                self.crowd_motion_prediction_stamped_rt.crowd_motion_prediction
            )

            # Publish a twist ROS message:
            cmd_vel_publisher.publish(cmd_vel_msg)
            crowd_prediction_publisher.publish(crowd_prediction_msg)

            rate.sleep()

    def update(self):
        raise NotImplementedError

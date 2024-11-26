import math
import threading
import rospy

import cv_bridge
import message_filters
import gazebo_msgs.msg
import sensor_msgs.msg

import tf2_ros

import labrob_crowd_navigation_msgs.msg
import labrob_crowd_navigation_utils.utils

import pal_detection_msgs.msg

class CrowdPerceptionCamera:

    def __init__(self):
        self.controller_frequency = 50 # [Hz]

        self.data_lock = threading.Lock()

        self.robot_configuration = labrob_crowd_navigation_utils.utils.Configuration(
            0.0, 0.0, 0.0
        )

        # Setting up reference frames:
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'

        # Setting up TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        person_detections_sub = message_filters.Subscriber(
            '/labrob_crowd_navigation/person_detector/detections',
            pal_detection_msgs.msg.Detections2d
        )

        depth_image_raw_sub = message_filters.Subscriber(
            '/xtion/depth_registered/image_raw',
            sensor_msgs.msg.Image
        )

        detections_depth_sync = message_filters.TimeSynchronizer(
            [person_detections_sub, depth_image_raw_sub],
            10
        )

        detections_depth_sync.registerCallback(
            self.person_detections_and_depth_callback
        )

        self.cv_bridge = cv_bridge.CvBridge()



    def person_detections_and_depth_callback(
            self,
            person_detections_msg,
            depth_image_msg
    ):
        depth_frame = self.cv_bridge.imgmsg_to_cv2(depth_image_msg, '32FC1')
        for detection in person_detections_msg.detections:
            rospy.loginfo(
                'Person detected at ({}, {}) at distance {}.'.format(
                    detection.x,
                    detection.y,
                    depth_frame[
                        int(detection.y + detection.height / 2.0),
                        int(detection.x + detection.width / 2.0)
                    ]
                )
            )

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

            crowd_motion_prediction = \
                labrob_crowd_navigation_utils.utils.CrowdMotionPrediction()
            
            # TODO: fill crowd_motion_prediction using detection data.
            # [ ] Use person detections to update measurements for EKF;
            # [ ] Use EKF to predict the motion of the persons (i.e., fill crowd_motion_prediction);
            # [x] Send crowd_motion_prediction to ROS topic.

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

    crowd_prediction_manager = CrowdPerceptionCamera()
    crowd_prediction_manager.start()

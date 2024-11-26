import math

import labrob_crowd_navigation_msgs.msg
import geometry_msgs.msg

class Configuration:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    
    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.theta)
    
    @staticmethod
    def set_from_tf_transform(transform):
        
        x = transform.transform.translation.x
        y = transform.transform.translation.y
        q = transform.transform.rotation
        theta = math.atan2(
          2.0 * (q.w * q.z + q.x * q.y),
          1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        config = Configuration(x, y, theta)
        return config

class UnicycleControlInput:
    def __init__(self, driving_velocity, steering_velocity):
        self.driving_velocity = driving_velocity
        self.steering_velocity = steering_velocity
    
    @staticmethod
    def to_twist_message(unicycle_control_input):
        cmd_vel_msg = geometry_msgs.msg.Twist()
        cmd_vel_msg.linear.x = unicycle_control_input.driving_velocity
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = unicycle_control_input.steering_velocity
        return cmd_vel_msg

class CrowdMotionPrediction:
    def __init__(self):
        self.motion_predictions = []
        self.size = 0
    
    def append(self, motion_prediction):
        self.motion_predictions.append(motion_prediction)
        self.size += 1

    @staticmethod
    def to_message(crowd_motion_prediction):
        crowd_motion_prediction_msg = \
            labrob_crowd_navigation_msgs.msg.CrowdMotionPrediction()
        for motion_prediction in crowd_motion_prediction.motion_predictions:
            crowd_motion_prediction_msg.motion_predictions.append(
                MotionPrediction.to_message(motion_prediction)
            )
        return crowd_motion_prediction_msg
    
    @staticmethod
    def from_message(crowd_motion_prediction_msg):
        crowd_motion_prediction = CrowdMotionPrediction()
        for motion_prediction_msg in \
            crowd_motion_prediction_msg.motion_predictions:
            crowd_motion_prediction.append(
                MotionPrediction.from_message(motion_prediction_msg)
            )
        return crowd_motion_prediction

class CrowdMotionPredictionStamped:
    def __init__(self, time, frame_id, crowd_motion_prediction):
        self.time = time
        self.frame_id = frame_id
        self.crowd_motion_prediction = crowd_motion_prediction

    @staticmethod
    def to_message(crowd_motion_prediction_stamped):
        crowd_motion_prediction_stamped_msg = \
            labrob_crowd_navigation_msgs.msg.CrowdMotionPredictionStamped()
        crowd_motion_prediction_stamped_msg.header.stamp = \
            crowd_motion_prediction_stamped.time
        crowd_motion_prediction_stamped_msg.header.frame_id = \
            crowd_motion_prediction_stamped.frame_id
        crowd_motion_prediction_stamped_msg.crowd_motion_prediction = \
            CrowdMotionPrediction.to_message(
                crowd_motion_prediction_stamped.crowd_motion_prediction
              )
        return crowd_motion_prediction_stamped_msg

    @staticmethod
    def from_message(crowd_motion_prediction_stamped_msg):
        return CrowdMotionPredictionStamped(
            crowd_motion_prediction_stamped_msg.header.stamp,
            crowd_motion_prediction_stamped_msg.header.frame_id,
            CrowdMotionPrediction.from_message(
                crowd_motion_prediction_stamped_msg.crowd_motion_prediction
            )
        )

class LaserScan:
    def __init__(
        self,
        time, frame_id,
        angle_min, angle_max, angle_increment,
        time_increment,
        scan_time,
        range_min, range_max,
        ranges, intensities):
        self.time            = time
        self.frame_id        = frame_id
        self.angle_min       = angle_min
        self.angle_max       = angle_max
        self.angle_increment = angle_increment
        self.range_min       = range_min
        self.range_max       = range_max
        self.ranges          = ranges
        self.intensities     = intensities

    @staticmethod
    def from_message(laser_scan_msg):
        return LaserScan(
            laser_scan_msg.header.stamp.to_sec(),
            laser_scan_msg.header.frame_id,
            laser_scan_msg.angle_min,
            laser_scan_msg.angle_max,
            laser_scan_msg.angle_increment,
            laser_scan_msg.time_increment,
            laser_scan_msg.scan_time,
            laser_scan_msg.range_min,
            laser_scan_msg.range_max,
            laser_scan_msg.ranges,
            laser_scan_msg.intensities
        )


class MotionPrediction:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
    
    @staticmethod
    def to_message(motion_prediction):
        return labrob_crowd_navigation_msgs.msg.MotionPrediction(
            Position.to_message(motion_prediction.position),
            Velocity.to_message(motion_prediction.velocity)
        )

    @staticmethod
    def from_message(motion_prediction_msg):
        return MotionPrediction(
            Position.from_message(motion_prediction_msg.position),
            Velocity.from_message(motion_prediction_msg.velocity)
        )

class Position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)

    @staticmethod
    def to_message(position):
        return geometry_msgs.msg.Point(position.x, position.y, position.z)
    
    @staticmethod
    def from_message(position_msg):
        return Position(position_msg.x, position_msg.y, position_msg.z)

class Velocity:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)
    
    @staticmethod
    def to_message(velocity):
        return geometry_msgs.msg.Vector3(velocity.x, velocity.y, velocity.z)
    
    @staticmethod
    def from_message(velocity_msg):
        return Velocity(velocity_msg.x, velocity_msg.y, velocity_msg.z)

class TiagoVelocity:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.theta)

def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))
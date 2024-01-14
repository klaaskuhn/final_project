#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint
from rclpy.node import Node
from tf_transformations import euler_from_quaternion


class PositionController(Node):

    def __init__(self):
        super().__init__(node_name='position_controller')

        self.thrust_pub = self.create_publisher(ActuatorSetpoint,
                                                'thrust_setpoint', 1)
        self.position_setpoint_sub = self.create_subscription(
            PointStamped, '~/setpoint', self.on_position_setpoint, 1)
        self.setpoint = Point()
        self.setpoint_timed_out = True
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                 'vision_pose_cov',
                                                 self.on_pose, 1)
        self.timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)
        self.pose_counter = 0
        self.time_before = self.get_clock().now()
        self.x_error_before = 0.0
        self.y_error_before = 0.0
        self.z_error_before = 0.0
        self.setpoint_before = None

    def on_setpoint_timeout(self):
        self.timeout_timer.cancel()
        self.get_logger().warn('setpoint timed out. waiting for new setpoints.')
        self.setpoint_timed_out = True

    def on_position_setpoint(self, msg: PointStamped):
        self.timeout_timer.reset()
        if self.setpoint_timed_out:
            self.get_logger().info('Setpoint received! Getting back to work.')
        self.setpoint_timed_out = False
        self.setpoint = msg.point

    def on_pose(self, msg: PoseWithCovarianceStamped):
        if self.setpoint_timed_out:
            return
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.apply_control(position, yaw)

    def apply_control(self, position: Point, yaw: float):
        now = self.get_clock().now()
        dt = (now-self.time_before).nanoseconds*1e-9
        x_error = self.setpoint.x - position.x
        y_error = self.setpoint.y - position.y
        z_error = self.setpoint.z - position.z

        x_derror = (x_error-self.x_error_before)/dt
        y_derror = (y_error-self.y_error_before)/dt
        z_derror = (z_error-self.z_error_before)/dt

        Kp = 3.0      #1
        Kd = 1.8    #0.5

        if self.setpoint_before != self.setpoint:
            Kd = 0

        x = Kp * x_error + Kd * x_derror
        y = Kp * y_error + Kd * y_derror
        z = Kp * z_error + Kd * z_derror

        self.time_before = now
        self.x_error_before = x_error
        self.y_error_before = y_error
        self.z_error_before = z_error
        self.setpoint_before = self.setpoint

        msg = ActuatorSetpoint()
        msg.header.stamp = now.to_msg()
        msg.x = math.cos(-yaw) * x - math.sin(-yaw) * y
        msg.x = min(0.5, max(-0.5, msg.x))
        msg.y = math.sin(-yaw) * x + math.cos(-yaw) * y
        msg.y = min(0.5, max(-0.5, msg.y))
        msg.z = z

        self.thrust_pub.publish(msg)


def main():
    rclpy.init()
    node = PositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

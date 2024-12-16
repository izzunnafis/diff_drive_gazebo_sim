#! /usr/bin/env python3

"""
    Author: Izzun Nafis
    Email: izzun1412nafis@gmail.com
    Ref: https://github.com/gurselturkeri/ros2_diff_drive_robot/tree/main
"""


import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class _Getch:
    """
    Gets a single character from standard input. 
    Does not echo to the screen.
    """

    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty
        import sys

    def __call__(self):
        import sys
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


class TeleopPublisher(Node):

    def __init__(self):
        super().__init__('teleop_node')
        self.left_motor_publisher = self.create_publisher(Float64MultiArray, '/left_wheel_controller/commands', 1)
        self.right_motor_publisher = self.create_publisher(Float64MultiArray, '/right_wheel_controller/commands', 1)
        self.vel_cmd_publisher = self.create_publisher(Twist, "/input_commands", 1)
        timer_period = 0.1
        self.timer = self.create_timer(
            timer_period, self.velocity_publish_event)

        self.left_motor_msg = Float64MultiArray()
        self.right_motor_msg = Float64MultiArray()
        self.input_command_msg = Twist()
        
        self.v = 0.0
        self.w = 0.0

    def set_vel(self, v, w):
        self.v = v
        self.w = w
                
    def velocity_publish_event(self):
        # Differential drive kinematics
        wheel_base = 1.0  # Distance between the wheels
        wheel_radius = 0.1  # Radius of the wheels

        # Calculate the wheel velocities
        w_left = (2 * self.v - self.w * wheel_base) / (2 * wheel_radius) #to wheel angular vel
        w_right = (2 * self.v + self.w * wheel_base) / (2 * wheel_radius)
        
        # Set the motor messages
        self.left_motor_msg.data = [w_left]
        self.right_motor_msg.data = [w_right]
        self.input_command_msg.linear.x = self.v
        self.input_command_msg.angular.z = self.w        
        
        self.left_motor_publisher.publish(self.left_motor_msg)
        self.right_motor_publisher.publish(self.right_motor_msg)
        self.vel_cmd_publisher.publish(self.input_command_msg)
        
        self.v /= 1.5
        self.w /= 1.5
        
        if abs(self.v) < 0.001:
            self.v = 0.0
        if abs(self.w) < 0.001:
            self.w = 0.0

def main(args=None):
    rclpy.init(args=args)
    getch = _Getch()
    publish_node = TeleopPublisher()
    thread = threading.Thread(target=rclpy.spin,
                              args=(publish_node, ), daemon=True)
    # Thread for node's timer callback
    thread.start()
    
    v = 0.0
    w = 0.0
    publish_node.get_logger().info("\n\tw: increment linear velocity by 0.5,\n\
        s: decrement linear velocity by 0.5,\n\
            max linear velocity is 2.5\n\
        a: increment angular velocity by 0.3,\n\
        d: decrement angular velocity by 0.3,\n\
            max angular velocity is 0.3\n\
        space: zero velocity command,\n\
        q: QUIT")
    
    try:
        while (rclpy.ok()):
            key_in = getch()  
            v = publish_node.v
            w = publish_node.w          
            if key_in == "w":
                v += 0.5
            elif key_in == "s":
                v -= 0.5
            elif key_in == "a":
                w += 0.3
            elif key_in == "d":
                w -= 0.3
            elif key_in == " ":
                v = 0.0
                w = 0.0
            elif key_in == "q":
                break
            
            if abs(v) > 2.5:
                v = 2.5 if v > 0 else -2.5
            if abs(w) > 1.5:
                w = 1.5 if w > 0 else -1.5

            publish_node.set_vel(v, w)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
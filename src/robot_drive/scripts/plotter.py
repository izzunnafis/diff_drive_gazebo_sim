import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from control_msgs.msg import DynamicJointState
from gazebo_msgs.msg import ModelStates
import matplotlib.pyplot as plt
import numpy as np

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        self.odom = []
        self.commands = []
        
        self.pose_command_x = 0.0
        self.pose_command_y = 0.0
        self.pose_command_theta = 0.0
        self.command_freq = 10.0
        
        self.pose_odom_x = 0.0
        self.pose_odom_y = 0.0
        self.pose_odom_theta = 0.0
        self.odom_freq = 10.0

        self.create_subscription(Twist, '/input_commands', self.input_pose_callback, 10)
        self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.odom_pose_callback, 10)
        
        self.wheel_base = 1.0  # Distance between the wheels
        self.wheel_radius = 0.1 # Radius of the wheels

        self.timer = self.create_timer(1.0, self.plot_data)
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

    def input_pose_callback(self, msg):
        self.pose_command_theta += msg.angular.z/self.odom_freq
        self.pose_command_x += msg.linear.x/self.command_freq * np.cos(self.pose_command_theta)
        self.pose_command_y += msg.linear.x/self.command_freq * np.sin(self.pose_command_theta)
        
        x = self.pose_command_x
        y = self.pose_command_y
        self.commands.append((x, y))  # Update current_pose with the latest position
        
    def odom_pose_callback(self, msg):
        vl = msg.interface_values[0].values[0] * self.wheel_radius
        vr = msg.interface_values[1].values[0] * self.wheel_radius
                
        v = (vr + vl) / 2.0
        w = (vr - vl) / self.wheel_base
        
        self.pose_odom_theta += w/self.odom_freq
        self.pose_odom_x += v/self.odom_freq * np.cos(self.pose_odom_theta)
        self.pose_odom_y += v/self.odom_freq * np.sin(self.pose_odom_theta)
        
        x = self.pose_odom_x
        y = self.pose_odom_y
        self.odom.append((x, y))
        
    def plot_data(self):        
        self.ax.clear()  # Clear previous plot
        
        if len(self.commands) > 0:
            x, y = zip(*self.commands)
            self.ax.plot(x, y, label='Command', color='blue')

        if len(self.odom) > 0:
            x, y = zip(*self.odom)
            self.ax.plot(x, y, label='Odometry', color='red')

        self.ax.legend()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Command and Odometry Visualization')
        self.ax.grid(True)
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        
        plt.draw()
        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    visualization_node = VisualizationNode()
    rclpy.spin(visualization_node)
    visualization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
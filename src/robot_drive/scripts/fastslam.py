import numpy as np
import random
import copy
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from control_msgs.msg import DynamicJointState
import numpy as np

from lib.robot import Robot
from lib.motion_model import MotionModel
from lib.measurement_model import MeasurementModel
from lib.utils import absolute2relative, relative2absolute, degree2radian

import time

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        
        self.wheel_base = 1.0  # Distance between the wheels
        self.wheel_radius = 0.1 # Radius of the wheels
        self.odom_freq = 10.0

        x, y, theta = 0.0, 0.0, 90.0
        self.pure_odom_x = x
        self.pure_odom_y = y
        self.pure_odom_theta = theta
        
        self.odom = []
        
        grid_size = 30 * 10
        world_grid = np.ones((grid_size, grid_size)) * 0.5
        self.R = Robot(x, y, degree2radian(theta), world_grid)
        self.estimated_R = self.R
        
        self.NUMBER_OF_PARTICLES = 8
        self.p = [None] * self.NUMBER_OF_PARTICLES
        for i in range(self.NUMBER_OF_PARTICLES):
            self.p[i] = Robot(x, y, degree2radian(theta), copy.deepcopy(world_grid))
            
        
        self.motion_model = MotionModel()
        self.measurement_model = MeasurementModel()
        
        self.curr_odo = self.prev_odo = self.R.get_state()

        self.create_subscription(LaserScan, '/scan', self.scan_callback, 5)
        self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.odom_pose_update, 5)
        
        self.step = 0
        self.update_time_count = 0
        
        
    def scan_callback(self, msg):
        time_start = time.time()
        self.curr_odo = self.R.get_state()
        
        z_star, free_grid_star, occupy_grid_star = self.R.process_beam(msg.ranges)
        free_grid_offset_star = absolute2relative(free_grid_star, self.curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, self.curr_odo)
        
        w = np.zeros(self.NUMBER_OF_PARTICLES)
        
        for i in range(self.NUMBER_OF_PARTICLES):
            prev_pose = self.p[i].get_state()
            x, y, theta = self.motion_model.sample_motion_model(self.prev_odo, self.curr_odo, prev_pose)
            self.p[i].set_states(x, y, theta)
            self.p[i].update_trajectory()
            
            z, _, _ = self.p[i].process_beam(msg.ranges)
            w[i] = self.measurement_model.measurement_model(z_star, z)
            
            curr_pose = self.p[i].get_state()
            free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(np.int32)
            occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(np.int32)
            self.p[i].update_occupancy_grid(free_grid, occupy_grid)
        
        w = w / np.sum(w)
        best_id = np.argsort(w)[-1]
        print(w)
        
        self.estimated_R = copy.deepcopy(self.p[best_id])
        
        
        new_p = [None] * self.NUMBER_OF_PARTICLES
        J_inv = 1 / self.NUMBER_OF_PARTICLES
        r = random.random() * J_inv
        c = w[0]
        
        i = 0
        for j in range(self.NUMBER_OF_PARTICLES):
            U = r + j * J_inv
            while (U > c):
                i += 1
                c += w[i]
            new_p[j] = copy.deepcopy(self.p[i])
        
        self.p = new_p
        self.prev_odo = self.curr_odo
        
        self.step += 1
        if self.step % 10 == 0:
            visualize(self, self.R, self.p, self.estimated_R, self.step, visualize=True, save=False)
        print("time end: ", time.time() - time_start)
        self.update_time_count = time.time() - time_start
        
    def odom_pose_update(self, msg):
        vl = msg.interface_values[0].values[0] * self.wheel_radius
        vr = msg.interface_values[1].values[0] * self.wheel_radius
                
        v = (vr + vl) / 2.0
        w = (vr - vl) / self.wheel_base
        
        self.pose_odom_x = self.R.get_state()[0]
        self.pose_odom_y = self.R.get_state()[1]
        self.pose_odom_theta = self.R.get_state()[2]
        
        if self.update_time_count > 0.1:
            update_freq = 1.0 / self.update_time_count
        else:
            update_freq = self.odom_freq
        
        # Pure odom
        self.pure_odom_theta += w/update_freq
        self.pure_odom_x += v/update_freq * np.cos(self.pose_odom_theta)
        self.pure_odom_y += v/update_freq * np.sin(self.pose_odom_theta)
        self.odom.append([self.pure_odom_x, self.pure_odom_y])
        
        # From particle
        [x, y, theta] = self.R.get_state()
        theta += w/update_freq
        x += v/update_freq * np.cos(theta) * 10.0
        y += v/update_freq * np.sin(theta) * 10.0
        
        self.R.set_states(x, y, theta)
        self.R.update_trajectory()
        
def visualize(self, robot, particles, best_particle, step, visualize=False, save=False):
    title = "FastSLAM"
    output_path = "fastslam"
    ax1.clear()
    fig.suptitle("{}\n\n number of particles:{}, step:{}".format(title, len(particles), step + 1))
    ax1.set_title("Estimated by Particles")
    ax1.axis("off")

    grid_size = best_particle.grid_size
    ax1.set_xlim(0, grid_size[1])
    ax1.set_ylim(0, grid_size[0])
        
    grid_size = robot.grid_size

    # draw map
    world_map = 1 - best_particle.grid
    ax1.imshow(world_map, cmap='gray')

    # draw trajectory
    estimated_path = np.array(best_particle.trajectory)
    ax1.plot(estimated_path[:, 0]+150, estimated_path[:, 1]+150, "g")
    
    # draw particles position
    for p in particles:
        ax1.plot(p.x+150, p.y+150, "bo", markersize=1)

    
    ax2.clear()
    if len(self.odom) > 0:
        x, y = zip(*self.odom)
        ax2.plot(x, y, label='Odometry', color='red')

    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Pure Odometry Visualization')
    ax2.grid(True)
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(-15, 15)


    if save:
        if step % 10 == 0:
            plt.savefig('{}_{}.png'.format(output_path, step), bbox_inches='tight')

    if visualize:
        plt.draw()
        plt.pause(0.01)

        


if __name__ == "__main__":
    rclpy.init()
    slam_node = SlamNode()
    rclpy.spin(slam_node)
    slam_node.destroy_node()
    rclpy.shutdown()
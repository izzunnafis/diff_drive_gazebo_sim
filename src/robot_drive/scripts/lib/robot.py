import random
import numpy as np

from .utils import *


class Robot(object):
    def __init__(self, x, y, theta, grid, sense_noise=None):
        # initialize robot pose
        self.x = x
        self.y = y
        self.theta = theta
        self.trajectory = []

        # map that robot navigates in
        # for particles, it is a map with prior probability
        self.grid = grid
        self.grid_size = self.grid.shape

        # probability for updating occupancy map
        self.prior_prob = 0.5
        self.occupy_prob = 0.9
        self.free_prob = 0.35

        # sensing noise for trun robot measurement
        # self.sense_noise = sense_noise if sense_noise is not None else 0.0

        # parameters for beam range sensor
        self.num_sensors = 30
        self.lidar_theta = np.arange(0, self.num_sensors) * (2 * np.pi / self.num_sensors) + np.pi / self.num_sensors
        self.lidar_length = 12
        self.lidar_range = 12

    def set_states(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def get_state(self):
        return (self.x, self.y, self.theta)
    
    def update_trajectory(self):
        self.trajectory.append([self.x, self.y])

    def process_beam(self, ranges):
        lidar_src = np.array([[self.x] * self.num_sensors, [self.y] * self.num_sensors])
        lidar_theta  = np.arange(0, self.num_sensors) * (2 * np.pi / self.num_sensors) + np.pi / self.num_sensors + self.theta - np.pi
        
        lidar_rel_dest = np.zeros((2, self.num_sensors))
        for i in range(self.num_sensors):
            lidar_rel_dest[0, i] = ranges[i] * np.cos(lidar_theta[i]) * 10
            lidar_rel_dest[1, i] = ranges[i] * np.sin(lidar_theta[i]) * 10
            
        lidar_rel_dest = np.clip(lidar_rel_dest, -self.lidar_range * 10, self.lidar_range * 10)
        lidar_dest = lidar_rel_dest + lidar_src
        
        beams = [None] * self.num_sensors
        for i in range(self.num_sensors):
            x1, y1 = lidar_src[0, i], lidar_src[1, i]
            x2, y2 = lidar_dest[0, i], lidar_dest[1, i]
            beams[i] = bresenham(x1 + self.grid_size[0]//2, y1 + self.grid_size[1]//2,
                                 x2 + self.grid_size[0]//2, y2 + self.grid_size[1]//2,
                                 self.grid_size[0], self.grid_size[1])

        measurements, free_grid, occupy_grid = self.ray_casting(beams)
        # measurements = np.clip(measurements + np.random.normal(0.0, self.sense_noise, self.num_sensors), 0.0, self.radar_range)
        
        return measurements, free_grid, occupy_grid

    def ray_casting(self, beams):
        loc = np.array([self.x + 150, self.y + 150])
        measurements = [self.lidar_range] * self.num_sensors
        free_grid, occupy_grid = [], []
        
        for i, beam in enumerate(beams):
            if len(beam) == 0:
                continue
            dist = np.linalg.norm(beam - loc, axis=1)
            beam = np.array(beam)
                        
            if dist[-1] <= self.lidar_range * 10 - 0.5:
                occupy_grid.append(list(beam[-1]))
                free_grid.extend(list(beam[:-1]))
                measurements[i] = dist[-1]
            else:
                free_grid.extend(list(beam))
                
        return measurements, free_grid, occupy_grid
    
    def update_occupancy_grid(self, free_grid, occupy_grid):
        mask1 = np.logical_and(0 < free_grid[:, 0], free_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < free_grid[:, 1], free_grid[:, 1] < self.grid_size[0])
        free_grid = free_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(False)
        l = prob2logodds(self.grid[free_grid[:, 1], free_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[free_grid[:, 1], free_grid[:, 0]] = logodds2prob(l)

        mask1 = np.logical_and(0 < occupy_grid[:, 0], occupy_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < occupy_grid[:, 1], occupy_grid[:, 1] < self.grid_size[0])
        occupy_grid = occupy_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(True)
        l = prob2logodds(self.grid[occupy_grid[:, 1], occupy_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[occupy_grid[:, 1], occupy_grid[:, 0]] = logodds2prob(l)
    
    def inverse_sensing_model(self, occupy):
        if occupy:
            return self.occupy_prob
        else:
            return self.free_prob

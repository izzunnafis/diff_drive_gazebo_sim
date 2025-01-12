import numpy as np

from .utils import normalDistribution

class MeasurementModel(object):
    def __init__(self):
        self.p_hit = 0.95
        self.p_short = 0.02
        self.p_max = 0.01
        self.p_rand = 0.02
        self.sigma_hit = 0.2
        self.lambda_short = 0.15
        self.lidar_range = 12
        self.eta_sort = 1

    def measurement_model(self, z_star, z):
        z_star, z = np.array(z_star), np.array(z)

        # probability of measuring correct range with local measurement noise
        prob_hit = normalDistribution(z - z_star, np.power(self.sigma_hit, 2))

        # probability of hitting unexpected objects
        prob_short = self.eta_sort * self.lambda_short * np.exp(-self.lambda_short * z)
        prob_short[np.greater(z, z_star)] = 0

        # probability of not hitting anything or failures
        prob_max = np.zeros_like(z)
        prob_max[z == self.lidar_range] = 1

        # probability of random measurements
        prob_rand = 1 / self.lidar_range

        # total probability (p_hit + p_shot + p_max + p_rand = 1)
        prob = self.p_hit * prob_hit + self.p_short * prob_short + self.p_max * prob_max + self.p_rand * prob_rand
        prob = np.prod(prob)

        return prob

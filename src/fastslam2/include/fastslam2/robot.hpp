#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <random>
#include <opencv4/opencv2/opencv.hpp>
#include "fastslam2/utils.hpp"

class Robot {
public:
    Robot(double x, double y, double theta, const cv::Mat& grid, double sense_noise = 0.0)
        : x(x), y(y), theta(theta), grid(grid), sense_noise(sense_noise) {
        grid_size = {grid.rows, grid.cols};
        
        prior_prob = 0.5;
        occupy_prob = 0.9;
        free_prob = 0.35;
        num_sensors = 30;
        lidar_length = 12;
        lidar_range = 12;
        for (int i = 0; i < num_sensors; ++i) {
            lidar_theta.push_back(i * (2 * M_PI / num_sensors) + M_PI / num_sensors);
        }
    }

    void set_states(double x, double y, double theta) {
        this->x = x;
        this->y = y;
        this->theta = wrapAngle(theta);
    }

    std::array<double, 3> get_state() const {
        return {x, y, theta};
    }

    cv::Mat get_grid() const {
        return grid;
    }

    void update_trajectory() {
        trajectory.push_back({x, y});
    }

    std::tuple<std::vector<double>, std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> process_beam(const std::vector<float>& ranges) {
        std::vector<Eigen::Vector2d> lidar_src(num_sensors, Eigen::Vector2d(x, y));
        std::vector<double> lidar_theta_adjusted(num_sensors);

        for (int i = 0; i < num_sensors; ++i) {
            lidar_theta_adjusted[i] = i * (2 * M_PI / num_sensors) + M_PI / num_sensors + theta - M_PI;
        }

        std::vector<Eigen::Vector2d> lidar_rel_dest(num_sensors);
        for (int i = 0; i < num_sensors; ++i) {
            lidar_rel_dest[i][0] = ranges[i] * std::cos(lidar_theta_adjusted[i]) * 10;
            lidar_rel_dest[i][1] = ranges[i] * std::sin(lidar_theta_adjusted[i]) * 10;
        }

        for (auto& dest : lidar_rel_dest) {
            dest[0] = std::clamp(dest[0], -lidar_range * 10.0, lidar_range * 10.0);
            dest[1] = std::clamp(dest[1], -lidar_range * 10.0, lidar_range * 10.0);
        }

        std::vector<Eigen::Vector2d> lidar_dest(num_sensors);
        for (int i = 0; i < num_sensors; ++i) {
            lidar_dest[i][0] = lidar_rel_dest[i][0] + lidar_src[i][0];
            lidar_dest[i][1] = lidar_rel_dest[i][1] + lidar_src[i][1];
        }

        std::vector<std::vector<std::array<int, 2>>> beams(num_sensors);
        for (int i = 0; i < num_sensors; ++i) {
            int x1 = static_cast<int>(lidar_src[i][0] + grid_size[0] / 2);
            int y1 = static_cast<int>(lidar_src[i][1] + grid_size[1] / 2);
            int x2 = static_cast<int>(lidar_dest[i][0] + grid_size[0] / 2);
            int y2 = static_cast<int>(lidar_dest[i][1] + grid_size[1] / 2);
            beams[i] = bresenham(x1, y1, x2, y2, grid_size[0], grid_size[1]);
        }

        return ray_casting(beams);
    }

    std::tuple<std::vector<double>, std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> ray_casting(const std::vector<std::vector<std::array<int, 2>>>& beams) {
        Eigen::Vector2d loc(x + 150, y + 150);
        std::vector<double> measurements(num_sensors, lidar_range);
        std::vector<Eigen::Vector2d> free_grid, occupy_grid;

        for (int i = 0; i < num_sensors; ++i) {
            const auto& beam = beams[i];
            if (beam.empty()) continue;

            std::vector<double> dist(beam.size());
            for (size_t j = 0; j < beam.size(); ++j) {
                dist[j] = (Eigen::Vector2d(beam[j][0], beam[j][1]) - loc).norm();
            }

            if (dist.back() <= lidar_range * 10 - 0.5) {
                occupy_grid.emplace_back(beam.back()[0], beam.back()[1]);
                for (size_t j = 0; j < beam.size() - 1; ++j) {
                    free_grid.emplace_back(beam[j][0], beam[j][1]);
                }
                measurements[i] = dist.back();
            } else {
                for (const auto& point : beam) {
                    free_grid.emplace_back(point[0], point[1]);
                }
            }
        }

        return {measurements, free_grid, occupy_grid};
    }


    void update_occupancy_grid(const std::vector<Eigen::Vector2d>& free_grid, const std::vector<Eigen::Vector2d>& occupy_grid) {
        auto update_grid = [&](const std::vector<Eigen::Vector2d>& grid_points, double inverse_prob) {
            for (const auto& point : grid_points) {
                int x = static_cast<int>(point[0]);
                int y = static_cast<int>(point[1]);
                if (x > 0 && x < grid_size[1] && y > 0 && y < grid_size[0]) {
                    double l = prob2logodds(grid.at<double>(y, x)) + prob2logodds(inverse_prob) - prob2logodds(prior_prob);
                    grid.at<double>(y, x) = logodds2prob(l);
                }
            }
        };

        update_grid(free_grid, inverse_sensing_model(false));
        update_grid(occupy_grid, inverse_sensing_model(true));
    }

private:
    double x, y, theta;
    cv::Mat grid;
    std::array<int, 2> grid_size;
    double prior_prob, occupy_prob, free_prob;
    double sense_noise;
    int num_sensors;
    std::vector<double> lidar_theta;
    int lidar_length, lidar_range;
    std::vector<std::array<double, 2>> trajectory;

    double inverse_sensing_model(bool occupy) const {
        return occupy ? occupy_prob : free_prob;
    }
};

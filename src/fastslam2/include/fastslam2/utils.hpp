#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <cmath>
#include <utility>
#include <vector>

// Bresenhams Line Generation Algorithm
// ref: https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/


std::vector<std::array<int, 2>> bresenham(int x1, int y1, int x2, int y2, int w, int h) {
    x1 = static_cast<int>(x1);
    y1 = static_cast<int>(y1);
    x2 = static_cast<int>(x2);
    y2 = static_cast<int>(y2);
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);

    bool steep = false;
    if (dx <= dy) {
        steep = true;
        std::swap(x1, y1);
        std::swap(x2, y2);
        std::swap(dx, dy);
    }

    int pk = 2 * dy - dx;
    std::vector<std::array<int, 2>> loc;

    for (int i = 0; i <= dx; ++i) {
        if ((x1 < 0 || y1 < 0) || (steep == false && (x1 >= h || y1 >= w)) || (steep == true && (x1 >= w || y1 >= h))) {
            break;
        }

        if (steep == false) {
            loc.emplace_back(std::array<int, 2>{x1, y1});
        } else {
            loc.emplace_back(std::array<int, 2>{y1, x1});
        }

        if (x1 < x2) {
            x1 = x1 + 1;
        } else {
            x1 = x1 - 1;
        }

        if (pk < 0) {
            pk = pk + 2 * dy;
        } else {
            if (y1 < y2) {
                y1 = y1 + 1;
            } else {
                y1 = y1 - 1;
            }
            pk = pk + 2 * dy - 2 * dx;
        }
    }

    return loc;
}


inline double wrapAngle(double radian) {
    radian = radian - 2 * M_PI * std::floor((radian + M_PI) / (2 * M_PI));
    return radian;
}

inline double degree2radian(double degree) {
    return degree / 180.0 * M_PI;
}

inline double prob2logodds(double prob) {
    return std::log(prob / (1 - prob + 1e-15));
}

inline double logodds2prob(double logodds) {
    return 1 - 1 / (1 + std::exp(logodds) + 1e-15);
}

inline double normalDistribution(double mean, double variance) {
    return std::exp(-(std::pow(mean, 2) / variance / 2.0) / std::sqrt(2.0 * M_PI * variance));
}

inline std::pair<Eigen::Matrix2d, Eigen::Matrix2d> create_rotation_matrix(double theta) {
    Eigen::Matrix2d R;
    R << std::cos(theta), -std::sin(theta),
            std::sin(theta), std::cos(theta);
    Eigen::Matrix2d R_inv = R.inverse();
    return std::make_pair(R, R_inv);
}

inline Eigen::Vector2d absolute2relative(const Eigen::Vector2d& position, const Eigen::Vector3d& states) {
    double x = states(0), y = states(1), theta = states(2);
    Eigen::Vector2d pose(x, y);
    
    auto [R, R_inv] = create_rotation_matrix(theta);
    Eigen::Vector2d pos = position - pose - Eigen::Vector2d(150.0, 150.0);
    pos = pos.transpose() * R_inv;
    
    return pos;
}

inline Eigen::Vector2d relative2absolute(const Eigen::Vector2d& position, const Eigen::Vector3d& states) {
    double x = states(0), y = states(1), theta = states(2);
    Eigen::Vector2d pose(x, y);
    
    auto [R, R_inv] = create_rotation_matrix(theta);
    Eigen::Vector2d pos = position.transpose() * R;
    pos = pos + pose;
    
    return pos + Eigen::Vector2d(150.0, 150.0);
}

#endif // UTILS_HPP
// Iterative Closest Point Algorithm
// Ref: https://github.com/AtsushiSakai/PythonRobotics/blob/53eae53b5a78a08b7ce4c6ffeed727c1d6a0ab2e/SLAM/iterative_closest_point/iterative_closest_point.py

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>
#include <numeric>

const double EPS = 0.0001;
const int MAX_ITER = 100;
const bool show_animation = false;

using namespace Eigen;

Matrix3d update_homogeneous_matrix(const Matrix3d& Hin, const Matrix2d& R, const Vector2d& T) {
    Matrix3d H = Matrix3d::Identity();
    H.block<2, 2>(0, 0) = R;
    H.block<2, 1>(0, 2) = T;
    return Hin * H;
}

std::pair<VectorXi, double> nearest_neighbor_association(const MatrixXd& prev_points, const MatrixXd& curr_points) {
    int n = curr_points.cols();
    int m = prev_points.cols();
    VectorXi indexes(n);
    double total_error = 0.0;

    for (int i = 0; i < n; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int min_index = -1;
        for (int j = 0; j < m; ++j) {
            double dist = (curr_points.col(i) - prev_points.col(j)).norm();
            if (dist < min_dist) {
                min_dist = dist;
                min_index = j;
            }
        }
        indexes(i) = min_index;
        total_error += min_dist;
    }

    return std::make_pair(indexes, total_error);
}

std::pair<Matrix2d, Vector2d> svd_motion_estimation(const MatrixXd& previous_points, const MatrixXd& current_points) {
    Vector2d pm = previous_points.rowwise().mean();
    Vector2d cm = current_points.rowwise().mean();

    MatrixXd p_shift = previous_points.colwise() - pm;
    MatrixXd c_shift = current_points.colwise() - cm;

    Matrix2d W = c_shift * p_shift.transpose();
    JacobiSVD<MatrixXd> svd(W, ComputeThinU | ComputeThinV);
    Matrix2d R = svd.matrixU() * svd.matrixV().transpose();
    Vector2d t = pm - R * cm;

    return std::make_pair(R, t);
}

Vector3d icp_matching(const MatrixXd& edges, MatrixXd scan, const Vector3d& pose) {
    if (scan.cols() < 5 || edges.cols() < scan.cols()) {
        return Vector3d::Zero();
    }

    // Remove duplicate columns
    std::vector<int> unique_indices;
    for (int i = 0; i < scan.cols(); ++i) {
        bool is_unique = true;
        for (int j = 0; j < i; ++j) {
            if ((scan.col(i) - scan.col(j)).norm() < EPS) {
                is_unique = false;
                break;
            }
        }
        if (is_unique) {
            unique_indices.push_back(i);
        }
    }
    MatrixXd unique_scan(2, unique_indices.size());
    for (size_t i = 0; i < unique_indices.size(); ++i) {
        unique_scan.col(i) = scan.col(unique_indices[i]);
    }
    scan = unique_scan;

    // Transpose edges and scan to match the implementation of algorithm
    MatrixXd edges_t = edges.transpose();
    MatrixXd scan_t = scan.transpose();

    Matrix3d H = Matrix3d::Identity();
    double dError = std::numeric_limits<double>::infinity();
    double preError = std::numeric_limits<double>::infinity();
    int count = 0;

    while (dError >= EPS) {
        count++;

        auto [indexes, total_error] = nearest_neighbor_association(edges_t, scan_t);
        MatrixXd edges_matched(2, scan_t.cols());
        for (int i = 0; i < scan_t.cols(); ++i) {
            edges_matched.col(i) = edges_t.col(indexes(i));
        }

        // Perform RANSAC
        double min_error = std::numeric_limits<double>::infinity();
        Matrix2d best_Rt;
        Vector2d best_Tt;
        for (int i = 0; i < 15; ++i) {
            std::vector<int> sample(scan_t.cols());
            std::iota(sample.begin(), sample.end(), 0);
            std::random_shuffle(sample.begin(), sample.end());
            sample.resize(5);

            MatrixXd sample_edges(2, 5);
            MatrixXd sample_scan(2, 5);
            for (int j = 0; j < 5; ++j) {
                sample_edges.col(j) = edges_matched.col(sample[j]);
                sample_scan.col(j) = scan_t.col(sample[j]);
            }

            auto [Rt, Tt] = svd_motion_estimation(sample_edges, sample_scan);
            MatrixXd temp_points = (Rt * scan_t).colwise() + Tt;
            auto [_, error] = nearest_neighbor_association(edges_matched, temp_points);
            if (error < min_error) {
                min_error = error;
                best_Rt = Rt;
                best_Tt = Tt;
            }
        }

        // Update current scan for iterative refinement
        scan_t = (best_Rt * scan_t).colwise() + best_Tt;

        dError = preError - total_error;
        preError = total_error;
        H = update_homogeneous_matrix(H, best_Rt, best_Tt);

        if (MAX_ITER <= count) {
            break;
        }
    }

    Matrix2d R = H.block<2, 2>(0, 0);
    Vector2d T = H.block<2, 1>(0, 2);

    if (std::abs(T(0)) > 5 || std::abs(T(1)) > 5) {
        return Vector3d::Zero();
    } else {
        double x = pose(0) + T(0);
        double y = pose(1) + T(1);
        double orientation = std::atan2(R(1, 0), R(0, 0));

        return Vector3d(x, y, orientation);
    }
}

// int main() {
//     // Example usage
//     MatrixXd edges(2, 10); // Replace with actual data
//     MatrixXd scan(2, 10);  // Replace with actual data
//     Vector3d pose(0, 0, 0); // Replace with actual data

//     Vector3d result = icp_matching(edges, scan, pose);
//     std::cout << "Result: " << result.transpose() << std::endl;

//     return 0;
// }
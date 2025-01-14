#ifndef FASTSLAM1_HPP
#define FASTSLAM1_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <control_msgs/msg/dynamic_joint_state.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <memory>
#include <ctime>

#include "fastslam1/robot.hpp"
#include "fastslam1/motion_model.hpp"
#include "fastslam1/measurement_model.hpp"
#include "fastslam1/utils.hpp"

#include "opencv4/opencv2/opencv.hpp"

#include <omp.h>

class SlamNode : public rclcpp::Node {
    public:
        SlamNode();
        ~SlamNode();

    private:
        void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

        void odom_pose_update(const control_msgs::msg::DynamicJointState::SharedPtr msg);

        double wheel_base;
        double wheel_radius;
        double odom_freq;
        double pure_odom_x;
        double pure_odom_y;
        double pure_odom_theta;
        double pose_odom_x;
        double pose_odom_y;
        double pose_odom_theta;

        double v_odom;
        double w_odom;

        int step;
        int move_forward;
        double update_time_count;
        int NUMBER_OF_PARTICLES;
        std::vector<std::shared_ptr<Robot>> p;
        std::shared_ptr<Robot> R;
        std::shared_ptr<Robot> estimated_R;
        std::shared_ptr<MotionModel> motion_model;
        std::shared_ptr<MeasurementModel> measurement_model;
        std::array<double, 3> curr_odo;
        std::array<double, 3> prev_odo;
        std::vector<std::vector<double>> odom;

        std::vector<double> temp_w;

        std::chrono::milliseconds sum_duration_exec;
        std::chrono::milliseconds sum_duration_iter;

        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription;
        rclcpp::Subscription<control_msgs::msg::DynamicJointState>::SharedPtr odom_subscription;
};

#endif // FASTSLAM1_HPP
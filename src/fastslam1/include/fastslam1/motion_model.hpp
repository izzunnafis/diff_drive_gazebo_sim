#ifndef MOTION_MODEL_HPP
#define MOTION_MODEL_HPP

#include <cmath>
#include <random>
#include <fastslam1/utils.hpp>

class MotionModel {
public:
    MotionModel() : alpha1(0.01), alpha2(0.01), alpha3(0.02), alpha4(0.02) {}

    std::tuple<double, double, double> sample_motion_model(const std::array<double, 3>& prev_odo, 
                                                            const std::array<double, 3>& curr_odo,  
                                                            const std::array<double, 3>& prev_pose, 
                                                            int move_forward) {

        double rot1 = 0; //assume that rot0 is always zero
        double trans = std::sqrt(std::pow(curr_odo[0] - prev_odo[0], 2) + std::pow(curr_odo[1] - prev_odo[1], 2)) * move_forward;
        double rot2 = curr_odo[2] - prev_odo[2] - rot1;
        rot2 = wrapAngle(rot2);

        rot1 = rot1 - normalDistributionGen(0, alpha1 * std::pow(rot1*10, 2) + alpha2 * std::pow(trans, 2));
        rot1 = wrapAngle(rot1);
        trans = trans - normalDistributionGen(0, alpha3 * std::pow(trans, 2) + alpha4 * (std::pow(rot1*10, 2) 
                + std::pow(rot2*10, 2)));
        rot2 = rot2 - normalDistributionGen(0, alpha1 * std::pow(rot2*10, 2) + alpha2 * std::pow(trans, 2));
        rot2 = wrapAngle(rot2);

        double x = prev_pose[0] + trans * std::cos(prev_pose[2] + rot1);
        double y = prev_pose[1] + trans * std::sin(prev_pose[2] + rot1);
        double theta = prev_pose[2] + rot1 + rot2;


        return std::make_tuple(x, y, theta);
    }

    double motion_model(const std::array<double, 3>& prev_odo, const std::array<double, 3>& curr_odo, const std::array<double, 3>& prev_pose, const std::array<double, 3>& curr_pose) {
        double rot1 = std::atan2(curr_odo[1] - prev_odo[1], curr_odo[0] - prev_odo[0]) - prev_odo[2];
        rot1 = wrapAngle(rot1);
        double trans = std::sqrt(std::pow(curr_odo[0] - prev_odo[0], 2) + std::pow(curr_odo[1] - prev_odo[1], 2));
        double rot2 = curr_odo[2] - prev_odo[2] - rot1;
        rot2 = wrapAngle(rot2);

        double rot1_prime = std::atan2(curr_pose[1] - prev_pose[1], curr_pose[0] - prev_pose[0]) - prev_pose[2];
        rot1_prime = wrapAngle(rot1_prime);
        double trans_prime = std::sqrt(std::pow(curr_pose[0] - prev_pose[0], 2) + std::pow(curr_pose[1] - prev_pose[1], 2));
        double rot2_prime = curr_pose[2] - prev_pose[2] - rot1_prime;
        rot2_prime = wrapAngle(rot2_prime);

        double p1 = normalDistributionGen(wrapAngle(rot1 - rot1_prime), alpha1 * std::pow(rot1_prime, 2) + alpha2 * std::pow(trans_prime, 2));
        double p2 = normalDistributionGen(trans - trans_prime, alpha3 * std::pow(trans_prime, 2) + alpha4 * (std::pow(rot1_prime, 2) + std::pow(rot2_prime, 2)));
        double p3 = normalDistributionGen(wrapAngle(rot2 - rot2_prime), alpha1 * std::pow(rot2_prime, 2) + alpha2 * std::pow(trans_prime, 2));

        return p1 * p2 * p3;
    }

private:
    double alpha1, alpha2, alpha3, alpha4;

    double normalDistributionGen(double mean, double stddev) {
        static std::default_random_engine generator;
        std::normal_distribution<double> distribution(mean, stddev);
        return distribution(generator);
    }
};

#endif // MOTION_MODEL_HPP
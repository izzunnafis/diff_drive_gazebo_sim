
#include <vector>
#include <cmath>
#include <algorithm>

class MeasurementModel {
public:
    MeasurementModel()
        : p_hit(0.8), p_short(0.05), p_max(0.1), p_rand(0.05),
            sigma_hit(2), lambda_short(0.15), lidar_range(12), eta_sort(1) {}

    double measurement_model(const std::vector<float>& z_now, const std::vector<double>& z_m) {
        std::vector<double> prob_hit(z_now.size());
        std::vector<double> prob_short(z_now.size());
        std::vector<double> prob_max(z_now.size(), 0);
        double prob_rand = 1.0 / (lidar_range*10);

        for (size_t i = 0; i < z_now.size(); ++i) {
            auto z_now_beam = z_now[i]*10.0;

            double diff = z_now_beam - z_m[i];
            prob_hit[i] = normalDistribution(diff, std::pow(sigma_hit, 2));

            if (z_now_beam <= z_m[i]) {
                prob_short[i] = eta_sort * lambda_short * std::exp(-lambda_short * z_m[i]);
            } else {
                prob_short[i] = 0;
            }

            if (z_now_beam == lidar_range) {
                prob_max[i] = 1;
            }
        }

        double prob = 1.0;
        for (size_t i = 0; i < z_m.size(); ++i) {
            prob *= p_hit * prob_hit[i] + p_short * prob_short[i] + p_max * prob_max[i] + p_rand * prob_rand;
        }

        return prob;
    }

private:
    double p_hit;
    double p_short;
    double p_max;
    double p_rand;
    double sigma_hit;
    double lambda_short;
    double lidar_range;
    double eta_sort;

    double normalDistribution(double x, double variance) {
        return (1.0 / std::sqrt(2.0 * M_PI * variance)) * std::exp(-0.5 * std::pow(x, 2) / variance);
    }
};

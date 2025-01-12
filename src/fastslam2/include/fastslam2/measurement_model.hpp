
#include <vector>
#include <cmath>
#include <algorithm>

class MeasurementModel {
public:
    MeasurementModel()
        : p_hit(0.95), p_short(0.02), p_max(0.01), p_rand(0.02),
            sigma_hit(0.2), lambda_short(0.15), lidar_range(12), eta_sort(1) {}

    double measurement_model(const std::vector<double>& z_star, const std::vector<double>& z) {
        std::vector<double> prob_hit(z.size());
        std::vector<double> prob_short(z.size());
        std::vector<double> prob_max(z.size(), 0);
        double prob_rand = 1.0 / lidar_range;

        for (size_t i = 0; i < z.size(); ++i) {
            double diff = z[i] - z_star[i];
            prob_hit[i] = normalDistribution(diff, std::pow(sigma_hit, 2));

            if (z[i] <= z_star[i]) {
                prob_short[i] = eta_sort * lambda_short * std::exp(-lambda_short * z[i]);
            } else {
                prob_short[i] = 0;
            }

            if (z[i] == lidar_range) {
                prob_max[i] = 1;
            }
        }

        double prob = 1.0;
        for (size_t i = 0; i < z.size(); ++i) {
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

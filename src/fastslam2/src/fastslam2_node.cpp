#include "fastslam2/fastslam2.hpp"

SlamNode::SlamNode() : Node("slam_node"), wheel_base(1.0), wheel_radius(0.1), odom_freq(10.0), step(0), update_time_count(0), move_forward(1) {
  double x = 0.0, y = 0.0, theta = 90.0;
  pure_odom_x = x;
  pure_odom_y = y;
  pure_odom_theta = degree2radian(theta);

  v_odom = 0.0;
  w_odom = 0.0;

  int grid_size = 30 * 10;
  cv::Mat world_grid(grid_size, grid_size, CV_64F, cv::Scalar(0.5));
  R = std::make_shared<Robot>(x, y, degree2radian(theta), world_grid);
  estimated_R = R;

  NUMBER_OF_PARTICLES = 8;
  NUMBER_OF_MODE_SAMPLES = 10;
  p.resize(NUMBER_OF_PARTICLES);
  for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
    p[i] = std::make_shared<Robot>(x, y, degree2radian(theta), world_grid);
  }

  motion_model = std::make_shared<MotionModel>();
  measurement_model = std::make_shared<MeasurementModel>();

  curr_odo = R->get_state();
  prev_odo = R->get_state();

  scan_subscription = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", 5, std::bind(&SlamNode::scan_callback, this, std::placeholders::_1));
  odom_subscription = this->create_subscription<control_msgs::msg::DynamicJointState>(
    "/dynamic_joint_states", 5, std::bind(&SlamNode::odom_pose_update, this, std::placeholders::_1));
}

void SlamNode::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  if (abs(v_odom) < 0.001 && abs(w_odom) < 0.001) {
    cv::Mat grid_image;
    cv::resize(estimated_R->get_grid(), grid_image, cv::Size(600, 600));

    for (const auto& particle : p) {
      auto state = particle->get_state();
      cv::circle(grid_image, cv::Point(state[0]*2 + grid_image.cols / 2, state[1]*2 + grid_image.rows / 2), 2, cv::Scalar(0.8), -1);
    }

    cv::flip(grid_image, grid_image, 0);
    cv::imshow("Occupancy Grid", 1-grid_image);
    cv::waitKey(1);

    return;
  }

  auto time_start = std::chrono::high_resolution_clock::now();
  curr_odo = R->get_state();

  auto [z_star, free_grid_star, occupy_grid_star] = R->process_beam(msg->ranges);
  Eigen::Vector3d curr_odo_eigen = Eigen::Vector3d(curr_odo[0], curr_odo[1], curr_odo[2]);
  
  auto free_grid_star_offset = free_grid_star;
  for (std::size_t i=0; i<free_grid_star.size(); i++) {
    free_grid_star_offset[i] = absolute2relative(free_grid_star[i], curr_odo_eigen);
  }

  auto occupy_grid_star_offset = occupy_grid_star;
  for (std::size_t i=0; i<occupy_grid_star.size(); i++) {
    occupy_grid_star_offset[i] = absolute2relative(occupy_grid_star[i], curr_odo_eigen);
  }
  
  std::vector<double> w(NUMBER_OF_PARTICLES, 1.0/NUMBER_OF_PARTICLES);


  for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
    auto prev_pose = p[i]->get_state();

    auto tmp_r = std::make_shared<Robot>(0.0, 0.0, 0.0, p[i]->get_grid());
    auto [x_guess, y_guess, theta_guess] = motion_model->sample_motion_model(prev_odo, curr_odo, prev_pose, move_forward);

    auto scan_guess = occupy_grid_star_offset; 
    for (std::size_t j=0; j<occupy_grid_star_offset.size(); j++) {
      scan_guess[i] = relative2absolute(occupy_grid_star_offset[j], Eigen::Vector3d(x_guess, y_guess, theta_guess));
    }

    cv::Mat tmp;
    cv::threshold(p[i]->get_grid(), tmp, 0.9, 1.0, cv::THRESH_BINARY);
    std::vector<cv::Point> edge;
    cv::findNonZero(tmp, edge);

    std::vector<Eigen::Vector2d> edge_eigen;
    for (const auto& pt : edge) {
      edge_eigen.emplace_back(pt.x, pt.y);
    }

    Eigen::MatrixXd edge_matrix(edge_eigen.size(), 2);
    for (std::size_t k = 0; k < edge_eigen.size(); ++k) {
      edge_matrix(k, 0) = edge_eigen[k](0);
      edge_matrix(k, 1) = edge_eigen[k](1);
    }

    Eigen::MatrixXd scan_matrix(scan_guess.size(), 2);
    for (std::size_t k = 0; k < scan_guess.size(); ++k) {
      scan_matrix(k, 0) = scan_guess[k](0);
      scan_matrix(k, 1) = scan_guess[k](1);
    }

    auto pose_hat = icp_matching(edge_matrix, scan_matrix, Eigen::Vector3d(x_guess, y_guess, theta_guess));

    // If the scan matching fails, the pose and the weights are computed according to the motion model
    if (!pose_hat.isZero()) {
      // Sample around the mode
      Eigen::MatrixXd samples(NUMBER_OF_MODE_SAMPLES, 3);
      Eigen::Vector3d mean = pose_hat;
      Eigen::Matrix3d cov;
      cov << 0.00005, 0, 0,
         0, 0.00005, 0,
         0, 0, 0.00001;
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<> d(0, 1);

      for (int j = 0; j < NUMBER_OF_MODE_SAMPLES; ++j) {
        samples.row(j) = mean + cov.llt().matrixL() * Eigen::Vector3d(d(gen), d(gen), d(gen));
      }

      // Compute gaussian proposal
      Eigen::VectorXd likelihoods(NUMBER_OF_MODE_SAMPLES);
      for (int j = 0; j < NUMBER_OF_MODE_SAMPLES; ++j) {
        std::array<double, 3> sample_array = {samples(j, 0), samples(j, 1), samples(j, 2)};
        auto motion_prob = motion_model->motion_model(prev_odo, curr_odo, prev_pose, sample_array, move_forward);

        double x = samples(j, 0);
        double y = samples(j, 1);
        double theta = samples(j, 2);
        tmp_r->set_states(x, y, theta);
        auto [z, _, __] = tmp_r->process_beam(msg->ranges);
        auto measurement_prob = measurement_model->measurement_model(z_star, z);

        likelihoods(j) = motion_prob * measurement_prob;
      }

      double eta = likelihoods.sum();
      if (eta > 0) {
        Eigen::Vector3d pose_mean = (samples.array().colwise() * likelihoods.array()).colwise().sum() / eta;

        Eigen::MatrixXd tmp = samples.rowwise() - pose_mean.transpose();
        // Eigen::Matrix3d pose_cov = (tmp.array().colwise() * likelihoods.array()).transpose() * tmp / eta;
        Eigen::Matrix3d pose_cov = tmp.transpose() * (tmp.array().colwise() * likelihoods.array()).matrix() / eta;


        // Sample new pose of the particle from the gaussian proposal
        Eigen::Vector3d new_pose = pose_mean + pose_cov.llt().matrixL() * Eigen::Vector3d(d(gen), d(gen), d(gen));
        p[i]->set_states(new_pose(0), new_pose(1), new_pose(2));
      }

      // Update weight
      w[i] *= eta;
      std::cout << "scan: " << w[i] << std::endl;
    } else {
      // Simulate a robot motion for each of these particles
      auto [x, y, theta] = motion_model->sample_motion_model(prev_odo, curr_odo, prev_pose, move_forward);
      p[i]->set_states(x, y, theta);

      // Calculate particle's weights depending on robot's measurement
      auto [z, _, __] = p[i]->process_beam(msg->ranges);
      w[i] *= measurement_model->measurement_model(z_star, z);
      std::cout << "not scan: " << w[i] << "; pose: " << pose_hat << std::endl;
    }

    p[i]->update_trajectory();

    auto curr_pose = p[i]->get_state();
    auto curr_pose_eigen = Eigen::Vector3d(curr_pose[0], curr_pose[1], curr_pose[2]);

    auto free_grid = free_grid_star_offset;
    for (std::size_t j=0; j<free_grid_star_offset.size(); j++) {
      free_grid[j] = relative2absolute(free_grid_star_offset[j], curr_pose_eigen);
    }

    auto occupy_grid = occupy_grid_star_offset;
    for (std::size_t j=0; j<occupy_grid_star_offset.size(); j++) {
      occupy_grid[j] = relative2absolute(occupy_grid_star_offset[j], curr_pose_eigen);
    }

    p[i]->update_occupancy_grid(free_grid, occupy_grid);
  }

  double sum_w = std::accumulate(w.begin(), w.end(), 0.0);
  std::transform(w.begin(), w.end(), w.begin(), [sum_w](double weight) { return weight / sum_w; });

  int best_id = std::distance(w.begin(), std::max_element(w.begin(), w.end()));
  estimated_R = std::make_shared<Robot>(*p[best_id]);

  // Adaptive resampling
  double N_eff = 1.0 / std::accumulate(w.begin(), w.end(), 0.0, [](double sum, double weight) { return sum + weight * weight; });
  if (N_eff < NUMBER_OF_PARTICLES / 2.0) {
    std::cout << "Resample!" << std::endl;
    // Resample the particles with a sample probability proportional to the importance weight
    // Use low variance sampling method
    std::vector<std::shared_ptr<Robot>> new_p(NUMBER_OF_PARTICLES);
    std::vector<double> new_w(NUMBER_OF_PARTICLES);
    double J_inv = 1.0 / NUMBER_OF_PARTICLES;
    double r = static_cast<double>(rand()) / RAND_MAX * J_inv;
    double c = w[0];

    int i = 0;
    for (int j = 0; j < NUMBER_OF_PARTICLES; ++j) {
      double U = r + j * J_inv;
      while (U > c) {
        i++;
        c += w[i];
      }
      new_p[j] = std::make_shared<Robot>(*p[i]);
      new_w[j] = w[i];
    }
    p = new_p;
    w = new_w;
  }

  prev_odo = curr_odo;

  step++;

  cv::Mat grid_image;
  cv::resize(estimated_R->get_grid(), grid_image, cv::Size(600, 600));

  for (const auto& particle : p) {
    auto state = particle->get_state();
    cv::circle(grid_image, cv::Point(state[0]*2 + grid_image.cols / 2, state[1]*2 + grid_image.rows / 2), 2, cv::Scalar(0.8), -1);
  }

  cv::flip(grid_image, grid_image, 0);
  cv::imshow("Occupancy Grid", 1-grid_image);
  cv::waitKey(1);
  
  auto time_end = std::chrono::high_resolution_clock::now();
  update_time_count = std::chrono::duration<double>(time_end - time_start).count();
  std::cout << "Update time: " << update_time_count << std::endl;
}

void SlamNode::odom_pose_update(const control_msgs::msg::DynamicJointState::SharedPtr msg) {
  double vl = msg->interface_values[0].values[0] * wheel_radius;
  double vr = msg->interface_values[1].values[0] * wheel_radius;

  v_odom = (vr + vl) / 2.0;
  w_odom = (vr - vl) / wheel_base;

  move_forward = (v_odom >= 0) ? 1 : -1;

  pose_odom_x = R->get_state()[0];
  pose_odom_y = R->get_state()[1];
  pose_odom_theta = R->get_state()[2];


  double update_freq = (update_time_count > 0.1) ? 1.0 / update_time_count : odom_freq;

  pure_odom_theta += w_odom / update_freq;
  pure_odom_x += v_odom / update_freq * std::cos(pose_odom_theta);
  pure_odom_y += v_odom / update_freq * std::sin(pose_odom_theta);

  pure_odom_theta = wrapAngle(pure_odom_theta);
  odom.push_back({pure_odom_x, pure_odom_y});

  auto [x, y, theta] = R->get_state();
  theta += w_odom / update_freq;
  x += v_odom / update_freq * std::cos(theta) * 10.0;
  y += v_odom / update_freq * std::sin(theta) * 10.0;

  R->set_states(x, y, theta);
  R->update_trajectory();

  auto [x_est, y_est, theta_est] = estimated_R->get_state();
  theta_est += w_odom / update_freq;
  x_est += v_odom / update_freq * std::cos(theta_est) * 10.0;
  y_est += v_odom / update_freq * std::sin(theta_est) * 10.0;

  estimated_R->set_states(x_est, y_est, theta_est);
  estimated_R->update_trajectory();
}



int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto slam_node = std::make_shared<SlamNode>();
  rclcpp::spin(slam_node);
  rclcpp::shutdown();
  return 0;
}
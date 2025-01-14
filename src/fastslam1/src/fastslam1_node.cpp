#include "fastslam1/fastslam1.hpp"

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

  NUMBER_OF_PARTICLES = 100;
  p.resize(NUMBER_OF_PARTICLES);
  for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
    p[i] = std::make_shared<Robot>(x, y, degree2radian(theta), world_grid);
  }

  motion_model = std::make_shared<MotionModel>();
  measurement_model = std::make_shared<MeasurementModel>();

  curr_odo = R->get_state();
  prev_odo = R->get_state();
  std::vector<double> temp_w(NUMBER_OF_PARTICLES, 0.0);


  scan_subscription = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", 3, std::bind(&SlamNode::scan_callback, this, std::placeholders::_1));
  odom_subscription = this->create_subscription<control_msgs::msg::DynamicJointState>(
    "/dynamic_joint_states", 3, std::bind(&SlamNode::odom_pose_update, this, std::placeholders::_1));
}

SlamNode::~SlamNode() {
  auto avg_duration_exec = sum_duration_exec.count() / step;
  auto avg_duration_iter = sum_duration_iter.count() / step;
  std::cout << "Average execution time: " << avg_duration_exec << " milliseconds" << std::endl;
  std::cout << "Average iteration time: " << avg_duration_iter << " milliseconds" << std::endl;

  cv::Mat grid_image;
  cv::Mat inverted_grid = 1 - estimated_R->get_grid();
  cv::resize(inverted_grid, grid_image, cv::Size(600, 600));

  grid_image.convertTo(grid_image, CV_8UC1, 255.0);
  cv::cvtColor(grid_image, grid_image, cv::COLOR_GRAY2BGR);

  std::vector<int> best_particles_indices(NUMBER_OF_PARTICLES);
  std::iota(best_particles_indices.begin(), best_particles_indices.end(), 0);
  // Corrected lambda to capture 'this'
  std::sort(best_particles_indices.begin(), best_particles_indices.end(), [this](int i, int j) { 
      return temp_w[i] > temp_w[j]; 
  });
  best_particles_indices.resize(5); // Keep only the 5 best particles

  for (int i = 0; i < 5; ++i) {
    cv::Mat particle_grid_image;
    cv::Mat inverted_particle_grid = 1 - p[best_particles_indices[i]]->get_grid();
    cv::resize(inverted_particle_grid, particle_grid_image, cv::Size(600, 600));

    particle_grid_image.convertTo(particle_grid_image, CV_8UC1, 255.0);
    cv::cvtColor(particle_grid_image, particle_grid_image, cv::COLOR_GRAY2BGR);

    auto particle_state = p[best_particles_indices[i]]->get_state();
    cv::circle(particle_grid_image, cv::Point(particle_state[0] * 2 + particle_grid_image.cols / 2, particle_state[1] * 2 + particle_grid_image.rows / 2), 2, cv::Scalar(255, 0, 0), -1);

    auto particle_trajectory = p[best_particles_indices[i]]->get_trajectory();
    for (const auto& point : particle_trajectory) {
      cv::circle(particle_grid_image, cv::Point(point[0] * 2 + particle_grid_image.cols / 2, point[1] * 2 + particle_grid_image.rows / 2), 1, cv::Scalar(0, 255, 0), -1);
    }

    cv::flip(particle_grid_image, particle_grid_image, 0);
    cv::imwrite("particle_" + std::to_string(best_particles_indices[i]) + ".png", particle_grid_image);
  }
}

void SlamNode::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  auto time_start = std::chrono::high_resolution_clock::now();
  curr_odo = R->get_state();
  
  std::vector<double> w(NUMBER_OF_PARTICLES, 0.0);

  auto start_time_exec = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for// default(none) shared(p, msg, z_star, free_grid_star_offset, occupy_grid_star_offset, w, curr_odo, prev_odo, move_forward)
  for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
    auto prev_pose = p[i]->get_state();
    auto [x, y, theta] = motion_model->sample_motion_model(prev_odo, curr_odo, prev_pose, move_forward); //Motion model
    p[i]->set_states(x, y, theta); //set to x_t
    p[i]->update_trajectory();

    auto z = p[i]->sense_beam(); //get the z from m_(-1) on pos x_t
    w[i] = measurement_model->measurement_model(msg->ranges, z); //Measurement model

    auto [free_grid, occupy_grid] = p[i]->process_beam(msg->ranges); //Ray casting process

    p[i]->update_occupancy_grid(free_grid, occupy_grid); //Update occupancy grid
  }
  auto end_time_exec = std::chrono::high_resolution_clock::now();
  auto duration_exec = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_exec - start_time_exec);
  sum_duration_exec += duration_exec;
  std::cout << "Execution time: " << duration_exec.count() << " milliseconds" << std::endl;

  double sum_w = std::accumulate(w.begin(), w.end(), 0.0);
  std::transform(w.begin(), w.end(), w.begin(), [sum_w](double weight) { return weight / sum_w; });

  int best_id = std::distance(w.begin(), std::max_element(w.begin(), w.end()));
  estimated_R = std::make_shared<Robot>(*p[best_id]);

  std::vector<std::shared_ptr<Robot>> new_p = p;
    
  double J_inv = 1.0 / NUMBER_OF_PARTICLES;
  double r = static_cast<double>(rand()) / RAND_MAX * J_inv;
  double c = w[0];

  int i = 0;
  for (int j = 0; j < NUMBER_OF_PARTICLES; ++j) {
    double U = r + j * J_inv;
    while (U > c) {
      ++i;
      c += w[i];
    }
    new_p[j]->x = p[i]->x;
    new_p[j]->y = p[i]->y;
    new_p[j]->theta = p[i]->theta;
  }

  p = new_p;
  temp_w = w;

  prev_odo = curr_odo;

  step++;

  cv::Mat grid_image;
  cv::Mat inverted_grid = 1 - estimated_R->get_grid();
  cv::resize(inverted_grid, grid_image, cv::Size(600, 600));

  grid_image.convertTo(grid_image, CV_8UC1, 255.0);
  cv::cvtColor(grid_image, grid_image, cv::COLOR_GRAY2BGR);

  for (const auto& particle : p) {
    auto state = particle->get_state();
    cv::circle(grid_image, cv::Point(state[0]*2 + grid_image.cols / 2, state[1]*2 + grid_image.rows / 2), 2, cv::Scalar(255, 0, 0), -1);
  }

  auto trajectory = estimated_R->get_trajectory();
  for (const auto& point : trajectory) {
    cv::circle(grid_image, cv::Point(point[0]*2 + grid_image.cols / 2, point[1]*2 + grid_image.rows / 2), 1, cv::Scalar(0, 255, 0), -1);
  }

  cv::flip(grid_image, grid_image, 0);
  cv::imshow("Occupancy Grid", grid_image);
  cv::waitKey(1);
  
  auto time_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
  sum_duration_iter += duration;
  std::cout << "Iteration time: " << duration.count() << " milliseconds" << std::endl;
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
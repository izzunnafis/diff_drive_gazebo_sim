cmake_minimum_required(VERSION 3.8)
project(fastslam2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)

find_package(image_transport REQUIRED )
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

add_executable(fastslam2_node src/fastslam2_node.cpp)

target_link_libraries(fastslam2_node
  ${OpenCV_LIBRARIES}
)

target_include_directories(fastslam2_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(fastslam2_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(fastslam2_node
  "rclcpp"
  "std_msgs"
  "sensor_msgs"
  "control_msgs"
  "cv_bridge"
  "image_transport"
  Eigen3
)

install(TARGETS fastslam2_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

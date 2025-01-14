cmake_minimum_required(VERSION 3.8)
project(fastslam1)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Add executable
add_executable(fastslam1_node src/fastslam1_node.cpp)

# Enable OpenMP
if(OpenMP_CXX_FOUND)
  target_compile_options(fastslam1_node PUBLIC ${OpenMP_CXX_FLAGS})
  target_link_libraries(fastslam1_node PUBLIC OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found. The build will proceed without OpenMP support.")
endif()

# Link OpenCV libraries
target_link_libraries(fastslam1_node PUBLIC ${OpenCV_LIBRARIES})

# Add include directories
target_include_directories(fastslam1_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Set C++ standard
target_compile_features(fastslam1_node PUBLIC cxx_std_17)

# Add dependencies
ament_target_dependencies(fastslam1_node PUBLIC
  rclcpp
  std_msgs
  sensor_msgs
  control_msgs
  cv_bridge
  image_transport
  Eigen3
)

# Install target
install(TARGETS fastslam1_node
  DESTINATION lib/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip linter warnings for license issues
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

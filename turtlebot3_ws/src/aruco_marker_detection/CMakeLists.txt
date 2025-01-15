cmake_minimum_required(VERSION 3.5)

# Set the project name
project(aruco_marker_detection)

# Find required ROS packages
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Declare Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python node
install(PROGRAMS
  src/aruco_detection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

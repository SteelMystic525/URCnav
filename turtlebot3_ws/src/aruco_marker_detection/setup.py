from setuptools import find_packages, setup

package_name = 'aruco_marker_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'opencv-python', 'rclpy', 'sensor_msgs'],
    zip_safe=True,
    maintainer='rishi',
    maintainer_email='suryanstripathi05@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detection_node = aruco_marker_detection.aruco_detection_node:main',
        ],
    },
)

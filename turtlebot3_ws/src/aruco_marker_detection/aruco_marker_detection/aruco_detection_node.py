import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R

class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__('aruco_detection_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # Initialize TF2 Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to image and camera info topics
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for marker coordinates
        self.marker_pub = self.create_publisher(String, '/aruco_marker_coordinates', 10)

        # Parameter for marker size (in meters)
        self.declare_parameter('marker_size', 0.05)

        # Timer to limit processing rate
        self.processing_timer = self.create_timer(0.1, self.process_images)

        # Internal image buffer
        self.latest_image = None

        self.get_logger().info("Aruco Detection Node Initialized.")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera info received and matrix initialized.")

    def image_callback(self, msg):
        # Buffer the latest image
        self.latest_image = msg

    def process_images(self):
        if self.latest_image is None:
            return

        if self.camera_matrix is None:
            self.get_logger().warn("Camera info not yet received. Skipping frame.")
            return

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')

            # Convert to grayscale
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Define ArUco dictionary and parameters
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            parameters = cv2.aruco.DetectorParameters()

            # Detect markers
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)

            if ids is not None:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

                # Get marker size from parameter
                marker_size = self.get_parameter('marker_size').get_parameter_value().double_value

                # Estimate pose of each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, self.camera_matrix, self.dist_coeffs)

                for i, marker_id in enumerate(ids):
                    tvec = tvecs[i][0]  # Translation vector
                    rvec = rvecs[i][0]  # Rotation vector

                    # Calculate Euclidean distance
                    distance = np.linalg.norm(tvec)

                    self.get_logger().info(
                        f"Marker ID: {marker_id[0]} | Position (x, y, z): {tvec} | Distance: {distance:.2f}m"
                    )

                    try:
                        # Look up transform from 'world' to 'camera_frame'
                        transform = self.tf_buffer.lookup_transform(
                            'world', 'camera_frame', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)
                        )

                        # Apply the transform to the marker's position
                        transformed_tvec = self.transform_marker_to_world(tvec, transform)

                        # Log and publish transformed coordinates
                        transformed_coords = f"Marker ID: {marker_id[0]}, World Position: {transformed_tvec}"
                        self.get_logger().info(transformed_coords)

                        self.marker_pub.publish(String(data=transformed_coords))

                    except tf2_ros.TransformException as e:
                        self.get_logger().warn(f"Transform failed: {e}")

                    # Draw axis for each marker
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

            # Display the image (optional for debugging)
            cv2.imshow("ArUco Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def transform_marker_to_world(self, tvec, transform):
        """
        Apply the transform to the marker position to convert it to the 'world' frame.
        """
        # Get the rotation and translation from the transform
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])
        quaternion = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # Apply the rotation and translation
        transformed_tvec = rotation_matrix @ tvec + translation
        return transformed_tvec

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Aruco Detection Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()





# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, CameraInfo
# import cv2
# from cv_bridge import CvBridge
# import numpy as np
# import tf2_ros

# class ArucoDetectionNode(Node):
#     def __init__(self):
#         super().__init__('aruco_detection_node')
#         self.bridge = CvBridge()
#         self.camera_matrix = None
#         self.dist_coeffs = None

#         # Initialize TF2 Buffer and Listener
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

#         # Subscribe to image and camera info topics
#         self.image_sub = self.create_subscription(
#             Image,
#             '/camera/image_raw',
#             self.image_callback,
#             10
#         )
#         self.camera_info_sub = self.create_subscription(
#             CameraInfo,
#             '/camera/camera_info',
#             self.camera_info_callback,
#             10
#         )

#         self.get_logger().info("Aruco Detection Node Initialized.")

#     def camera_info_callback(self, msg):
#         if self.camera_matrix is None:
#             self.camera_matrix = np.array(msg.k).reshape((3, 3))
#             self.dist_coeffs = np.array(msg.d)
#             self.get_logger().info("Camera info received and matrix initialized.")

#     def image_callback(self, msg):
#         if self.camera_matrix is None:
#             self.get_logger().warn("Camera info not yet received. Skipping frame.")
#             return

#         try:
#             # Convert ROS Image message to OpenCV image
#             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

#             # Convert to grayscale
#             gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

#             # Define ArUco dictionary and parameters
#             aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#             parameters = cv2.aruco.DetectorParameters()

#             # Detect markers
#             corners, ids, rejected = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)

#             if ids is not None:
#                 # Draw detected markers
#                 cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

#                 # Estimate pose of each marker
#                 rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.camera_matrix, self.dist_coeffs)

#                 for i, marker_id in enumerate(ids):
#                     # Display pose information
#                     tvec = tvecs[i][0]  # Translation vector
#                     rvec = rvecs[i][0]  # Rotation vector

#                     distance = np.linalg.norm(tvec)  # Calculate Euclidean distance

#                     self.get_logger().info(
#                         f"Marker ID: {marker_id[0]} | Position (x, y, z): {tvec} | Distance: {distance:.2f}m"
#                     )

#                     # Look up transform from 'world' to 'camera_frame'
#                     try:
#                         transform = self.tf_buffer.lookup_transform(
#                             'world', 'camera_frame', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)
#                         )

#                         # Apply the transform to the marker's position
#                         transformed_tvec = self.transform_marker_to_world(tvec, transform)
#                         self.get_logger().info(f"Transformed Marker Position: {transformed_tvec}")

#                     except tf2_ros.TransformException as e:
#                         self.get_logger().warn(f"Transform failed: {e}")

#                     # Draw axis for each marker
#                     cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

#             # Display the image (optional for debugging)
#             cv2.imshow("ArUco Detection", cv_image)
#             cv2.waitKey(1)

#         except Exception as e:
#             self.get_logger().error(f"Error processing image: {e}")

#     def transform_marker_to_world(self, tvec, transform):
#         """
#         Apply the transform to the marker position to convert it to the 'world' frame.
#         """
#         # Create a numpy array for the marker translation vector
#         marker_position = np.array(tvec)

#         # Get the translation part of the transform
#         transform_translation = np.array([transform.transform.translation.x, 
#                                           transform.transform.translation.y, 
#                                           transform.transform.translation.z])

#         # Combine the marker position with the transformation
#         transformed_position = marker_position + transform_translation

#         return transformed_position

# def main(args=None):
#     rclpy.init(args=args)
#     node = ArucoDetectionNode()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info("Shutting down Aruco Detection Node.")
#     finally:
#         cv2.destroyAllWindows()
#         node.destroy_node()
#         rclpy.shutdown()










# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2

# class ArucoDetectionNode(Node):
#     def __init__(self):
#         super().__init__('aruco_detection_node')
#         # Initialize CvBridge
#         self.bridge = CvBridge()
#         # Create a subscription to the /camera/image_raw topic
#         self.create_subscription(
#             Image,
#             '/camera/image_raw',
#             self.image_callback,
#             10
#         )
#         # Initialize the ArUco detector
#         dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#         parameters = cv2.aruco.DetectorParameters()
#         self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        
#         self.get_logger().info('Aruco Detection Node has been started. Subscribed to /camera/image_raw.')

#     def image_callback(self, msg):
#         self.get_logger().info('Frame received from /camera/image_raw.')
#         try:
#             # Convert ROS Image to OpenCV format
#             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
#             # Convert image to grayscale
#             gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
#             # Detect ArUco markers using the detector object
#             corners, ids, rejected = self.detector.detectMarkers(gray_image)
            
#             if ids is not None:
#                 self.get_logger().info(f'Detected {len(ids)} marker(s). IDs: {ids.flatten()}')
#                 # Draw markers on the image
#                 cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
#             else:
#                 self.get_logger().info('No ArUco markers detected.')
                
#             # Display the image with detections
#             cv2.imshow('ArUco Detection', cv_image)
#             cv2.waitKey(1)
            
#         except Exception as e:
#             self.get_logger().error(f'Error processing image: {str(e)}')

# def main(args=None):
#     rclpy.init(args=args)
#     node = ArucoDetectionNode()
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Node interrupted by user.')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()

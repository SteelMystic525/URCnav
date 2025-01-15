
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
import numpy as np

class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__('aruco_detection_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

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

        self.get_logger().info("Aruco Detection Node Initialized.")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera info received and matrix initialized.")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            self.get_logger().warn("Camera info not yet received. Skipping frame.")
            return

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

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

                # Estimate pose of each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.camera_matrix, self.dist_coeffs)

                for i, marker_id in enumerate(ids):
                    # Display pose information
                    tvec = tvecs[i][0]  # Translation vector
                    rvec = rvecs[i][0]  # Rotation vector

                    distance = np.linalg.norm(tvec)  # Calculate Euclidean distance

                    self.get_logger().info(
                        f"Marker ID: {marker_id[0]} | Position (x, y, z): {tvec} | Distance: {distance:.2f}m"
                    )

                    # Draw axis for each marker
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

            # Display the image (optional for debugging)
            cv2.imshow("ArUco Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Aruco Detection Node.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()









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
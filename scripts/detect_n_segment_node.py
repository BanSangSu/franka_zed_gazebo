#!/usr/bin/env python3

"""
Multi-Detector + SAM Integrated ROS Node
Supports: Florence-2, DINO, Grounding DINO, YOLOv11 + SAM Segmentation
"""

import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as RosImage, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sklearn.decomposition import PCA
import tf2_ros
import tf2_geometry_msgs

import numpy as np
import cv2

from detect_n_segment import MultiDetectorSAM 

# ==================== ROS NODE ====================

class Cube:
    def __init__(self, id, position, orientation, confidence, dimensions):
        self.id = id
        self.position = position
        self.orientation = orientation
        self.confidence = confidence
        self.dimensions = dimensions

class SamCubeDetector:
    def __init__(self):
        rospy.init_node('sam_cube_detector', anonymous=True)
        
        # Params
        # self.detector_type = rospy.get_param('~detector_type', 'yolov11')
        self.detector_type = rospy.get_param('~detector_type', 'florence2')
        self.yolo_weights = rospy.get_param('~yolo_weights', 'yolo11n.pt') 
        self.sam_checkpoint = rospy.get_param('~sam_checkpoint', 'sam_vit_b_01ec64.pth')
        self.sam_model_type = rospy.get_param('~sam_model_type', 'vit_b')
        self.prompt = rospy.get_param('~prompt', 'small cube') # For GroundingDINO/Florence
        self.target_classes = rospy.get_param('~target_classes', [])
        
        self.camera_frame = rospy.get_param('~camera_frame', 'static_zed2_left_camera_optical_frame')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.min_points_for_pca = rospy.get_param('~min_points_for_pca', 50)
        self.depth_scale = rospy.get_param('~depth_scale', 1.0)
        
        self.bridge = CvBridge()
        self.cubes = []
        self.camera_K = None
        self.camera_frame_id = None
        
        # Initialize Pipeline
        # detector_config = {"model_name": self.yolo_weights} if self.detector_type == 'yolov11' else {}
        detector_config = {"model_name": "microsoft/Florence-2-base"} if self.detector_type == 'florence2' else {}
        self.pipeline = MultiDetectorSAM(
            detector_type=self.detector_type,
            detector_config=detector_config,
            sam_checkpoint=self.sam_checkpoint,
            sam_model_type=self.sam_model_type
        )
        
        # # ROS Communication
        # image_topic = rospy.get_param('~image_topic', "/zed2/zed_node/rgb/image_rect_color")
        # camera_info_topic = rospy.get_param('~camera_info_topic', "/zed2/zed_node/rgb/camera_info")
        # depth_topic = rospy.get_param('~depth_topic', "/zed2/zed_node/depth/depth_registered")
        # ZED2 topics
        image_topic = rospy.get_param('~image_topic', "/static_zed2/zed_node/left/image_rect_color")
        camera_info_topic = rospy.get_param('~camera_info_topic', "/static_zed2/zed_node/left/camera_info")
        depth_topic = rospy.get_param('~depth_topic', "/static_zed2/zed_node/depth/depth_registered")
        
        # Subscribers
        image_sub = message_filters.Subscriber(image_topic, RosImage)
        camera_info_sub = message_filters.Subscriber(camera_info_topic, CameraInfo)
        depth_sub = message_filters.Subscriber(depth_topic, RosImage)
        
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, camera_info_sub, depth_sub], 
            10, 0.1
        )
        ts.registerCallback(self.detection_callback)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.cube_publishers = {}
        self.point_cloud_pub = rospy.Publisher('/sam_cube_detection/cubes_pointcloud', PointCloud2, queue_size=10)
        self.seg_image_pub = rospy.Publisher('/sam_cube_detection/segmentation_image', RosImage, queue_size=10)
        
        rospy.loginfo(f"SAM Cube Detector initialized with {self.detector_type}")

    def detection_callback(self, image_msg, camera_info_msg, depth_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            
            if self.camera_K is None:
                self.camera_K = np.array(camera_info_msg.K).reshape(3, 3)
                self.camera_frame_id = camera_info_msg.header.frame_id
                rospy.loginfo(f"Camera intrinsics initialized: fx={self.camera_K[0,0]:.2f}, fy={self.camera_K[1,1]:.2f}")
            
            try:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            except CvBridgeError as e:
                rospy.logerr(f"Depth conversion error: {e}")
                return
                
            # Run Detection + SAM
            # For YOLO, prompt is not used in the same way, but can be for filtering
            result = self.pipeline.detect_and_segment(
                cv_image, 
                prompt=self.prompt if self.detector_type != 'yolov11' else None,
                conf=0.25
            )
            
            self.visualize_and_publish(result, cv_image, image_msg.header)
            self.process_detections(result, cv_image, depth_image, image_msg.header)
            
        except Exception as e:
            rospy.logerr(f"Callback error: {e}")

    def visualize_and_publish(self, result, cv_image, header):
        vis_img = cv_image.copy()
        masks = result['segmentation']['masks']
        bboxes = result['detection']['bboxes']
        labels = result['detection']['labels']
        scores = result['detection']['scores']
        
        for i, mask in enumerate(masks):
            color = np.random.randint(0, 255, 3).tolist()
            vis_img[mask > 0] = vis_img[mask > 0] * 0.5 + np.array(color) * 0.5
            
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = labels[i] if i < len(labels) else "obj"
            score = scores[i] if i < len(scores) else 0.0
            cv2.putText(vis_img, f"{label} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        try:
            msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
            msg.header = header
            self.seg_image_pub.publish(msg)
        except: pass

    def process_detections(self, result, cv_image, depth_image, header):
        self.cubes = []
        masks = result['segmentation']['masks']
        scores = result['detection']['scores']
        
        img_h, img_w = cv_image.shape[:2]
        
        for idx, mask in enumerate(masks):
            # Mask is bool or uint8
            mask_binary = (mask > 0).astype(np.uint8)
            
            points_3d = self.extract_3d_points_from_mask(mask_binary, depth_image)
            
            if len(points_3d) < self.min_points_for_pca: 
                continue
            
            centroid, orientation, dims = self.compute_pca_features(points_3d)
            if centroid is None: 
                continue
            
            w_pos, w_orient = self.transform_to_world(centroid, orientation, self.camera_frame_id, self.world_frame)
            
            if w_pos is not None:
                confidence = scores[idx] if idx < len(scores) else 1.0
                cube = Cube(idx, w_pos, w_orient, confidence, dims)
                self.cubes.append(cube)
                rospy.loginfo(f"Cube {cube.id}: pos={w_pos}, conf={confidence:.2f}")
                
        self.publish_cubes(header)

    def extract_3d_points_from_mask(self, mask_binary, depth_image):
        y_coords, x_coords = np.where(mask_binary > 0)
        if len(x_coords) == 0: return np.empty((0, 3))
        
        fx, fy = self.camera_K[0, 0], self.camera_K[1, 1]
        cx, cy = self.camera_K[0, 2], self.camera_K[1, 2]
        
        depths = depth_image[y_coords, x_coords] * self.depth_scale
        
        # Valid depths
        valid = (depths > 0.1) & (depths < 10.0) & np.isfinite(depths)
        
        x_c = x_coords[valid]
        y_c = y_coords[valid]
        z_c = depths[valid]
        
        if len(z_c) == 0: return np.empty((0, 3))
        
        x_3d = (x_c - cx) * z_c / fx
        y_3d = (y_c - cy) * z_c / fy
        
        return np.stack([x_3d, y_3d, z_c], axis=1)

    def compute_pca_features(self, points):
        rospy.logdebug(f"Computing PCA for {len(points)} points")
        if len(points) < 3: return None, None, None
        
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        try:
            pca = PCA(n_components=3)
            pca.fit(centered_points)
        except Exception as e:
            rospy.logwarn(f"PCA failed: {e}")
            return None, None, None
            
        principal_axes = pca.components_
        if np.linalg.det(principal_axes) < 0:
            principal_axes[2] = -principal_axes[2]
            
        quaternion = self.rotation_matrix_to_quaternion(principal_axes)
        
        projected = centered_points @ principal_axes.T
        dimensions = 2 * np.std(projected, axis=0)
        
        return centroid, quaternion, dimensions

    def rotation_matrix_to_quaternion(self, R):
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])

    def transform_to_world(self, position, orientation, source_frame, target_frame):
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = rospy.Time(0)
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = position
            
            transformed_point = self.tf_buffer.transform(point_stamped, target_frame, rospy.Duration(1.0))
            world_position = np.array([transformed_point.point.x, transformed_point.point.y, transformed_point.point.z])
            
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            q_tf = transform.transform.rotation
            q_tf_arr = [q_tf.x, q_tf.y, q_tf.z, q_tf.w]
            
            world_orientation = self.quaternion_multiply(q_tf_arr, orientation)
            return world_position, world_orientation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logdebug(f"TF transform failed: {e}")
            return position, orientation

    def quaternion_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def publish_cubes(self, header):
        timestamp = rospy.Time.now()
        
        # Publish individual cube odometry
        for cube in self.cubes:
            topic_name = f"/sam_cube_detection/cube_{cube.id}_odom"
            if topic_name not in self.cube_publishers:
                self.cube_publishers[topic_name] = rospy.Publisher(topic_name, Odometry, queue_size=10, latch=True)
            
            odom = Odometry()
            odom.header.stamp = timestamp
            odom.header.frame_id = self.world_frame
            odom.child_frame_id = f"cube_{cube.id}"
            
            odom.pose.pose.position.x = cube.position[0]
            odom.pose.pose.position.y = cube.position[1]
            odom.pose.pose.position.z = cube.position[2]
            
            odom.pose.pose.orientation.x = cube.orientation[0]
            odom.pose.pose.orientation.y = cube.orientation[1]
            odom.pose.pose.orientation.z = cube.orientation[2]
            odom.pose.pose.orientation.w = cube.orientation[3]
            
            self.cube_publishers[topic_name].publish(odom)
        
        # PointCloud
        if len(self.cubes) > 0:
            points = [[c.position[0], c.position[1], c.position[2], c.confidence] for c in self.cubes]
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1),
            ]
            pc_msg = pc2.create_cloud(Header(stamp=timestamp, frame_id=self.world_frame), fields, points)
            self.point_cloud_pub.publish(pc_msg)

    def shutdown(self):
        rospy.loginfo("Shutting down SAM Cube Detector...")

def main():
    try:
        detector = SamCubeDetector()
        rospy.spin()
    except rospy.ROSInterruptException: pass
    except Exception as e: rospy.logerr(f"Fatal error: {e}")
    finally:
        if 'detector' in locals(): locals()['detector'].shutdown()

if __name__ == '__main__':
    main()

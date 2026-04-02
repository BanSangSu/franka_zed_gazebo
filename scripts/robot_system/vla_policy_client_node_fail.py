#!/usr/bin/env python3
### Have to fix websocket error 2026.03.27. We use websockets
import rospy
import message_filters
import numpy as np
import websocket
import msgpack
import msgpack_numpy
import threading

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Patch msgpack to handle numpy arrays seamlessly
msgpack_numpy.patch()

class VLAPolicyClient:
    def __init__(self):
        rospy.init_node('vla_policy_client_node', anonymous=True)

        # Parameters
        self.ws_url = rospy.get_param('~ws_url', 'ws://localhost:8000')
        self.task_description = rospy.get_param('~task_description', 'stack the small block on the table')
        self.rgb_topic = rospy.get_param('~rgb_topic', '/zedr/zed_node/left/image_rect_color')
        self.depth_topic = rospy.get_param('~depth_topic', '/zedr/zed_node/depth/depth_registered')
        self.joint_state_topic = rospy.get_param('~joint_state_topic', '/joint_states')
        self.command_topic = rospy.get_param('~command_topic', '/eff_joint_traj_controller/command')
        self.control_rate = rospy.get_param('~control_rate', 10.0) # Hz
        
        # Franka joint names in order
        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        self.bridge = CvBridge()
        self.latest_obs = None
        self.obs_lock = threading.Lock()

        # Connect to Policy WebSocket Server
        rospy.loginfo(f"Connecting to Diffusion Policy Server at {self.ws_url}...")
        try:
            self.ws = websocket.create_connection(self.ws_url, timeout=5.0)
            rospy.loginfo("✓ Connected to Policy Server.")
            self._send_init_bounds()
        except Exception as e:
            rospy.logerr(f"Failed to connect to WebSocket server: {e}")
            rospy.signal_shutdown("WebSocket connection failed.")
            return

        # Setup ROS Publishers
        self.cmd_pub = rospy.Publisher(self.command_topic, JointTrajectory, queue_size=1)

        # Setup Synchronized Subscribers
        self.sub_rgb = message_filters.Subscriber(self.rgb_topic, Image)
        self.sub_depth = message_filters.Subscriber(self.depth_topic, Image)
        self.sub_joints = message_filters.Subscriber(self.joint_state_topic, JointState)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_joints], 
            queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # Start control loop
        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)
        rospy.loginfo("✓ VLA Policy Client running.")

    def _send_init_bounds(self):
        """Send action bounds if required by the server init phase."""
        init_payload = {
            "action_low": np.full(7, -2.8973).astype(np.float64),  # Franka generic lower limits
            "action_high": np.full(7, 2.8973).astype(np.float64)   # Franka generic upper limits
        }
        # Assuming the policy_websocket uses msgpack unpacking
        self.ws.send(msgpack.packb(init_payload, use_bin_type=True))
        self.ws.recv() # Wait for dummy zero action response

    def sync_callback(self, rgb_msg, depth_msg, joint_msg):
        """Synchronize incoming sensor data and format it for the policy."""
        try:
            # Convert images to numpy arrays
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb_img = rgb_img[..., ::-1] # Convert BGR to RGB
            
            # Depth usually needs to be converted to float32 meters
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            # Extract 14-dim proprioception (7 positions + 7 velocities)
            pos, vel = [], []
            for name in self.joint_names:
                if name in joint_msg.name:
                    idx = joint_msg.name.index(name)
                    pos.append(joint_msg.position[idx])
                    vel.append(joint_msg.velocity[idx])
                else:
                    pos.append(0.0)
                    vel.append(0.0)
                    
            proprio = np.array(pos + vel, dtype=np.float64)

            # Store safely for the control thread
            with self.obs_lock:
                self.latest_obs = {
                    "primary_image": rgb_img,       # (H, W, 3) uint8 RGB
                    "depth_image": depth_img,       # (H, W) float
                    "proprio": proprio,             # (14,) float64
                    "task_description": self.task_description
                }

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")

    def control_loop(self, event):
        """Query the model and execute the received action chunk."""
        with self.obs_lock:
            if self.latest_obs is None:
                return
            obs_payload = self.latest_obs.copy()

        try:
            # Send observation to WebSocket server
            self.ws.send(msgpack.packb(obs_payload, use_bin_type=True))

            # Receive and unpack action response
            response = self.ws.recv()
            result = msgpack.unpackb(response, raw=False)
            
            if "actions" in result:
                action = np.array(result["actions"]) # Should be (7,) float64
                self.execute_action(action)
            else:
                rospy.logwarn("No 'actions' key in server response.")

        except Exception as e:
            rospy.logerr(f"Error communicating with policy server: {e}")

    def execute_action(self, action_array):
        """Publish the 7-DOF action to the robot's trajectory controller."""
        if len(action_array) != 7:
            rospy.logwarn(f"Expected action dim 7, got {len(action_array)}.")
            return

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        traj_msg.header.stamp = rospy.Time.now()

        point = JointTrajectoryPoint()
        point.positions = action_array.tolist()
        
        # Give the controller the exact time interval to reach this point based on rate
        point.time_from_start = rospy.Duration(1.0 / self.control_rate) 
        
        traj_msg.points.append(point)
        self.cmd_pub.publish(traj_msg)

    def shutdown(self):
        rospy.loginfo("Shutting down VLA client...")
        if hasattr(self, 'ws') and self.ws.connected:
            self.ws.close()

if __name__ == '__main__':
    try:
        client = VLAPolicyClient()
        rospy.on_shutdown(client.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
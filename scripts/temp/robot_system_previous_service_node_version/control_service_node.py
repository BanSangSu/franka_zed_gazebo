#!/usr/bin/env python3

import rospy
import moveit_commander
import actionlib

from franka_zed_gazebo.srv import ControlService, ControlServiceResponse, GripperService, GripperServiceResponse

# Try to import Franka Gripper Action (for force-controlled grasping)
try:
    from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
    HAS_FRANKA_GRIPPER = True
except ImportError:
    HAS_FRANKA_GRIPPER = False
    rospy.logwarn("franka_gripper messages not found. Using MoveIt gripper control (position only).")


class ControlServiceNode:
    def __init__(self):
        rospy.init_node('control_service_node')
        
        # Initialize moveit_commander
        moveit_commander.roscpp_initialize([])
        
        self.robot = moveit_commander.RobotCommander()
        
        # Get planning group name
        planning_group = rospy.get_param('~planning_group', 'panda_manipulator')
        self.move_group = moveit_commander.MoveGroupCommander(planning_group, wait_for_servers=30)
        
        # Initialize gripper group (MoveIt fallback)
        gripper_group_name = rospy.get_param('~gripper_group', 'panda_hand')
        self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, wait_for_servers=30)

        # # Configure execution parameters
        # self.move_group.set_max_velocity_scaling_factor(rospy.get_param('~max_velocity', 0.5))
        # self.move_group.set_max_acceleration_scaling_factor(rospy.get_param('~max_acceleration', 0.5))
        
        # Gripper parameters
        self.default_grasp_force = rospy.get_param('~default_grasp_force', 20.0)  # Increased to 60N
        self.grasp_epsilon_inner = rospy.get_param('~grasp_epsilon_inner', 0.02)  # Increased tolerance
        self.grasp_epsilon_outer = rospy.get_param('~grasp_epsilon_outer', 0.02)
        self.gripper_speed = rospy.get_param('~gripper_speed', 0.1)  # m/s
        self.max_gripper_width = rospy.get_param('~max_gripper_width', 0.08)
        
        # Enable force control by default if available
        self.use_force_control = rospy.get_param('~use_force_control', True)
        
        # Initialize Franka Gripper Action clients if available
        self.grasp_client = None
        self.move_client = None
        if HAS_FRANKA_GRIPPER:
            try:
                self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
                self.move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
                
                if self.grasp_client.wait_for_server(timeout=rospy.Duration(2.0)):
                    rospy.loginfo("Franka Gripper Action servers connected - Force control enabled")
                else:
                    rospy.logwarn("Franka Gripper Action servers not available. Using MoveIt fallback.")
                    self.grasp_client = None
                    self.move_client = None
            except Exception as e:
                rospy.logwarn(f"Failed to connect to Franka Gripper: {e}. Using MoveIt fallback.")
                self.grasp_client = None
                self.move_client = None
        
        # Create services
        self.service = rospy.Service('/control_service', ControlService, self.handle_control_request)
        self.gripper_service = rospy.Service('/gripper_service', GripperService, self.handle_gripper_request)
        
        rospy.loginfo(f"Control Service Node Ready for group: {planning_group} and gripper: {gripper_group_name}")

    def handle_control_request(self, req):
        """Service callback for trajectory execution requests"""
        response = ControlServiceResponse()
        
        try:
            if len(req.trajectory.joint_trajectory.points) == 0:
                response.success = False
                response.message = "Empty trajectory received"
                return response
            
            rospy.loginfo(f"Executing trajectory with {len(req.trajectory.joint_trajectory.points)} waypoints")
            
            # Execute the trajectory
            success = self.move_group.execute(req.trajectory, wait=True)
            
            if success:
                response.success = True
                response.message = "Trajectory executed successfully"
                rospy.loginfo(response.message)
            else:
                response.success = False
                response.message = "Trajectory execution failed"
                rospy.logwarn(response.message)
            
            # Stop any residual motion
            self.move_group.stop()
            
        except Exception as e:
            response.success = False
            response.message = f"Control error: {str(e)}"
            rospy.logerr(response.message)
            self.move_group.stop()
        
        return response

    def handle_gripper_request(self, req):
        """Service callback for gripper requests with force control"""
        response = GripperServiceResponse()
        
        try:
            if req.open:
                # --- OPEN GRIPPER ---
                response = self._open_gripper()
            else:
                # --- CLOSE/GRASP ---
                target_width = req.width if req.width > 0 else 0.0
                grasp_force = req.force if req.force > 0 else self.default_grasp_force
                response = self._grasp(target_width, grasp_force)
                
        except Exception as e:
            response.success = False
            response.message = f"Gripper error: {str(e)}"
            response.final_width = -1.0
            rospy.logerr(response.message)
        
        return response
    
    def _open_gripper(self):
        """Open the gripper to max width"""
        response = GripperServiceResponse()
        
        if self.move_client is not None:
            # Use Franka Gripper Move Action
            goal = MoveGoal()
            goal.width = self.max_gripper_width
            goal.speed = self.gripper_speed
            
            self.move_client.send_goal(goal)
            self.move_client.wait_for_result(rospy.Duration(5.0))
            
            result = self.move_client.get_result()
            if result and result.success:
                response.success = True
                response.message = "Gripper opened (Franka Action)"
            else:
                response.success = False
                response.message = "Gripper open failed"
        else:
            # Fallback to MoveIt position control
            joint_goal = self.gripper_group.get_current_joint_values()
            if len(joint_goal) >= 2:
                joint_goal[0] = 0.04  # Max per finger
                joint_goal[1] = 0.04
                self.gripper_group.go(joint_goal, wait=True)
                self.gripper_group.stop()
                response.success = True
                response.message = "Gripper opened (MoveIt)"
            else:
                self.gripper_group.set_named_target("open")
                self.gripper_group.go(wait=True)
                self.gripper_group.stop()
                response.success = True
                response.message = "Gripper opened (MoveIt named target)"
        
        # Get final width
        current_joints = self.gripper_group.get_current_joint_values()
        response.final_width = sum(current_joints) if len(current_joints) >= 2 else self.max_gripper_width
        
        rospy.loginfo(response.message)
        return response
    
    def _grasp(self, target_width, force):
        """
        Close the gripper to grasp an object.
        
        Uses MoveAction instead of GraspAction to avoid asymmetric contact issues in Gazebo.
        MoveAction simply moves to the target width without force sensing.
        
        Args:
            target_width: Target width between fingers (meters). 
            force: Grasping force in Newtons (not used with MoveAction, kept for API compatibility).
        """
        response = GripperServiceResponse()
        
        # === FORCE CONTROL MODE (GraspAction) - For real robot ===
        if self.use_force_control and self.grasp_client is not None:
            goal = GraspGoal()
            goal.width = target_width
            goal.epsilon.inner = self.grasp_epsilon_inner
            goal.epsilon.outer = self.grasp_epsilon_outer
            goal.speed = self.gripper_speed
            goal.force = force
            
            rospy.loginfo(f"Grasping with FORCE CONTROL: force={force}N, target_width={target_width}m")
            
            self.grasp_client.send_goal(goal)
            self.grasp_client.wait_for_result(rospy.Duration(10.0))
            
            result = self.grasp_client.get_result()
            current_joints = self.gripper_group.get_current_joint_values()
            response.final_width = sum(current_joints) if len(current_joints) >= 2 else 0.0
            
            if result and result.success:
                response.success = True
                response.message = f"Grasp succeeded (force={force}N, width={response.final_width:.4f}m)"
                rospy.loginfo(response.message)
            else:
                width_error = abs(response.final_width - target_width)
                if response.final_width < 0.005:
                    response.success = False
                    response.message = f"Grasp failed (no object: width={response.final_width:.4f}m)"
                    rospy.logwarn(response.message)
                elif width_error <= 0.02:
                    response.success = True
                    response.message = f"Grasp acceptable (width={response.final_width:.4f}m)"
                    rospy.loginfo(response.message)
                else:
                    response.success = False
                    response.message = f"Grasp rejected: width {response.final_width:.4f}m vs target {target_width:.4f}m"
                    rospy.logwarn(response.message)
        
        # === POSITION CONTROL MODE (MoveAction) - For Gazebo simulation ===
        elif self.move_client is not None:
            # Use the provided target_width (already has a small margin from orchestrator)
            # If target_width is 0 (not provided), fallback to 0.001 for a full close
            sim_target = target_width if target_width > 0 else 0.001
            
            goal = MoveGoal()
            goal.width = sim_target
            goal.speed = 0.05 # Closing speed
            
            rospy.loginfo(f"Grasping (Sim): Moving to {sim_target:.4f}m for controlled compression.")
            self.move_client.send_goal(goal)
            
            # 2. Wait and monitor - We wait until velocity is zero or timeout
            # In Gazebo, we want the fingers to keep PUSHING
            timeout_time = rospy.Time.now() + rospy.Duration(3.0)
            last_width = 1.0
            rate = rospy.Rate(5.0)  # 5 Hz (0.2s interval)
            
            while not rospy.is_shutdown() and rospy.Time.now() < timeout_time:
                current_joints = self.gripper_group.get_current_joint_values()
                current_width = sum(current_joints) if len(current_joints) >= 2 else 0.0
                
                # Check if movement stopped (reached object)
                if abs(current_width - last_width) < 0.0001:
                    # Stalled, likely hit object
                    break
                
                last_width = current_width
                rate.sleep()  # Maintains 5Hz frequency and handles is_shutdown correctly
            
            # Add a small extra delay to let physics settle
            rospy.sleep(0.5)
            
            # 3. Final verification
            response.final_width = last_width
            
            # If fingers closed almost completely, we probably missed the object
            if response.final_width < 0.005:
                response.success = False
                response.message = f"Grasp failed (closed completely: {response.final_width:.4f}m)"
                self.move_client.cancel_all_goals()
                rospy.logwarn(response.message)
            else:
                # WE DO NOT CANCEL THE GOAL HERE
                # Keeping the goal active maintains pressure in Gazebo
                response.success = True
                response.message = f"Grasped and holding at width={response.final_width:.4f}m"
                rospy.loginfo(response.message)
        else:
            # Fallback to MoveIt position control (NO force control)
            rospy.logwarn("Using MoveIt fallback - no force control available! Targeting near-zero for tight grip.")
            
            joint_goal = self.gripper_group.get_current_joint_values()
            if len(joint_goal) >= 2:
                # Target near-zero to ensure fingers keep pushing against the object
                # Simulation physics will prevent actual overlap
                joint_goal[0] = 0.001
                joint_goal[1] = 0.001
                self.gripper_group.go(joint_goal, wait=True)
                # Note: We don't call stop() immediately as it might clear the effort
            else:
                self.gripper_group.set_named_target("close")
                self.gripper_group.go(wait=True)
            
            # Check if we grasped something
            current_joints = self.gripper_group.get_current_joint_values()
            response.final_width = sum(current_joints) if len(current_joints) >= 2 else 0.0
            
            # If fingers closed completely (no object)
            if response.final_width < 0.002:
                response.success = False
                response.message = "Grasp failed (no object detected)"
                rospy.logwarn(response.message)
            else:
                response.success = True
                response.message = f"Gripper closed (MoveIt, width={response.final_width:.4f}m)"
                rospy.loginfo(response.message)
        
        return response


if __name__ == '__main__':
    try:
        node = ControlServiceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        moveit_commander.roscpp_shutdown()

#!/usr/bin/env python3
import rospy
import moveit_commander
import actionlib
import threading

from franka_zed_gazebo.msg import (
    ControlActionAction, ControlActionFeedback, ControlActionResult,
    GripperActionAction, GripperActionFeedback, GripperActionResult
)

try:
    from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
    HAS_FRANKA_GRIPPER = True
except ImportError:
    HAS_FRANKA_GRIPPER = False
    rospy.logwarn("franka_gripper not found. Using MoveIt fallback.")


class ControlActionServer:
    """
    Stateless action server for behavior tree integration.
    Provides atomic trajectory execution and gripper control actions.
    State management is delegated to the behavior tree.
    """
    
    def __init__(self):
        rospy.init_node('control_action_server')
        
        # Initialize moveit_commander
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        
        # Get planning groups
        planning_group = rospy.get_param('~planning_group', 'panda_manipulator')
        gripper_group_name = rospy.get_param('~gripper_group', 'panda_hand')
        
        self.move_group = moveit_commander.MoveGroupCommander(
            planning_group, wait_for_servers=30
        )
        self.gripper_group = moveit_commander.MoveGroupCommander(
            gripper_group_name, wait_for_servers=30
        )
        
        # Motion parameters - Match service node speeds
        self.max_velocity = rospy.get_param('~max_velocity', 0.5)
        self.max_acceleration = rospy.get_param('~max_acceleration', 0.5)
        self.move_group.set_max_velocity_scaling_factor(self.max_velocity)
        self.move_group.set_max_acceleration_scaling_factor(self.max_acceleration)
        
        # Gripper parameters
        self.default_grasp_force = rospy.get_param('~default_grasp_force', 20.0)
        self.grasp_epsilon_inner = rospy.get_param('~grasp_epsilon_inner', 0.02)
        self.grasp_epsilon_outer = rospy.get_param('~grasp_epsilon_outer', 0.02)
        self.gripper_speed = rospy.get_param('~gripper_speed', 0.1)
        self.max_gripper_width = rospy.get_param('~max_gripper_width', 0.08)
        self.use_force_control = rospy.get_param('~use_force_control', True)
        
        # Initialize Franka Gripper clients
        self._init_gripper_clients()
        
        # Create action servers with preempt callbacks
        self.control_server = actionlib.SimpleActionServer(
            '/control_action',
            ControlActionAction,
            execute_cb=self.execute_control_cb,
            auto_start=False
        )
        self.control_server.register_preempt_callback(self.control_preempt_cb)
        
        self.gripper_server = actionlib.SimpleActionServer(
            '/gripper_action',
            GripperActionAction,
            execute_cb=self.execute_gripper_cb,
            auto_start=False
        )
        self.gripper_server.register_preempt_callback(self.gripper_preempt_cb)
        
        self.control_server.start()
        self.gripper_server.start()
        
        rospy.loginfo(f"âœ“ Control Action Server: {planning_group} "
                      f"(vel={self.max_velocity}, acc={self.max_acceleration})")
        rospy.loginfo(f"âœ“ Gripper Action Server: {gripper_group_name} "
                      f"(force_control={self.use_force_control and self.grasp_client is not None})")
    
    def _init_gripper_clients(self):
        """Initialize Franka Gripper action clients"""
        self.grasp_client = None
        self.move_client = None
        
        if not HAS_FRANKA_GRIPPER:
            return
        
        try:
            franka_gripper_grasp = rospy.resolve_name('franka_gripper/grasp')
            franka_gripper_move = rospy.resolve_name('franka_gripper/move')
            self.grasp_client = actionlib.SimpleActionClient(
                franka_gripper_grasp, GraspAction
            )
            self.move_client = actionlib.SimpleActionClient(
                franka_gripper_move, MoveAction
            )
            
            if self.move_client.wait_for_server(timeout=rospy.Duration(2.0)):
                rospy.loginfo("âœ“ Franka Gripper servers connected")
            else:
                rospy.logwarn("âš  Franka Gripper unavailable. Using MoveIt.")
                self.grasp_client = None
                self.move_client = None
        except Exception as e:
            rospy.logwarn(f"âš  Gripper connection failed: {e}")
            self.grasp_client = None
            self.move_client = None
    
    # ========== CONTROL ACTION CALLBACKS ==========
    
    def control_preempt_cb(self):
        """Handle trajectory preemption gracefully"""
        rospy.logwarn("Control action preempted by behavior tree")
        self.move_group.stop()
    
    def execute_control_cb(self, goal):
        """
        Execute trajectory with real-time feedback.
        Stateless - each call is independent.
        """
        feedback = ControlActionFeedback()
        result = ControlActionResult()
        
        try:
            num_points = len(goal.trajectory.joint_trajectory.points)
            if num_points == 0:
                result.success = False
                result.message = "Empty trajectory"
                self.control_server.set_aborted(result)
                return
            
            rospy.loginfo(f"â–¶ Executing trajectory: {num_points} waypoints")
            
            # Initial feedback
            feedback.progress = 0.0
            feedback.status = "Starting execution"
            self.control_server.publish_feedback(feedback)
            
            # Execute with monitoring
            success = self._execute_with_monitoring(goal.trajectory, feedback)
            
            # Stop residual motion
            self.move_group.stop()
            
            # Final feedback
            feedback.progress = 1.0
            feedback.status = "Complete" if success else "Failed"
            self.control_server.publish_feedback(feedback)
            
            result.success = success
            result.message = "Executed" if success else "Failed"
            
            if success:
                rospy.loginfo(f"âœ“ {result.message}")
                self.control_server.set_succeeded(result)
            else:
                rospy.logwarn(f"âœ— {result.message}")
                self.control_server.set_aborted(result)
                
        except Exception as e:
            result.success = False
            result.message = f"Error: {str(e)}"
            rospy.logerr(result.message)
            self.move_group.stop()
            self.control_server.set_aborted(result)
    
    def _execute_with_monitoring(self, trajectory, feedback):
        """
        Execute trajectory with progress monitoring.
        Runs execution in thread to allow feedback publishing.
        """
        execution_result = {'success': False}
        
        def execute_trajectory():
            execution_result['success'] = self.move_group.execute(trajectory, wait=True)
        
        execution_thread = threading.Thread(target=execute_trajectory)
        execution_thread.start()
        
        # Monitor and publish feedback
        rate = rospy.Rate(5)  # 5 Hz
        start_time = rospy.Time.now()
        duration = trajectory.joint_trajectory.points[-1].time_from_start
        
        while execution_thread.is_alive() and not rospy.is_shutdown():
            # Check for preemption
            if self.control_server.is_preempt_requested():
                self.move_group.stop()
                execution_thread.join(timeout=1.0)
                return False
            
            # Estimate progress
            elapsed = (rospy.Time.now() - start_time).to_sec()
            progress = min(elapsed / duration.to_sec(), 0.95) if duration.to_sec() > 0 else 0.5
            
            feedback.progress = progress
            feedback.status = f"Executing ({int(progress*100)}%)"
            self.control_server.publish_feedback(feedback)
            
            rate.sleep()
        
        execution_thread.join()
        return execution_result['success']
    
    # ========== GRIPPER ACTION CALLBACKS ==========
    
    def gripper_preempt_cb(self):
        """Handle gripper preemption"""
        rospy.logwarn("Gripper action preempted by behavior tree")
        
        # Cancel ongoing gripper actions
        if self.move_client:
            self.move_client.cancel_all_goals()
        if self.grasp_client:
            self.grasp_client.cancel_all_goals()
    
    def execute_gripper_cb(self, goal):
        """
        Execute gripper action (open/close).
        Stateless - behavior tree manages gripper state.
        """
        feedback = GripperActionFeedback()
        result = GripperActionResult()
        
        try:
            if goal.open:
                success, message, final_width = self._open_gripper(feedback)
            else:
                target_width = goal.width if goal.width > 0 else 0.0
                grasp_force = goal.force if goal.force > 0 else self.default_grasp_force
                if goal.mode == "picking":
                    success, message, final_width = self._grasp(
                        target_width, grasp_force, feedback
                    )
                else:
                    success, message, final_width = self._grasp(
                        0.0, grasp_force, feedback
                    )

                    
            
            result.success = success
            result.message = message
            result.final_width = final_width
            
            if success:
                rospy.loginfo(f"âœ“ {message}")
                self.gripper_server.set_succeeded(result)
            else:
                rospy.logwarn(f"âœ— {message}")
                self.gripper_server.set_aborted(result)
                
        except Exception as e:
            result.success = False
            result.message = f"Error: {str(e)}"
            result.final_width = -1.0
            rospy.logerr(result.message)
            self.gripper_server.set_aborted(result)
    
    # ========== GRIPPER CONTROL METHODS ==========
    
    def _open_gripper(self, feedback):
        """Open gripper to maximum width"""
        feedback.status = "Opening gripper"
        feedback.progress = 0.0
        self.gripper_server.publish_feedback(feedback)
        
        if self.move_client is not None:
            # Franka Move Action
            goal = MoveGoal()
            goal.width = self.max_gripper_width
            goal.speed = self.gripper_speed
            
            self.move_client.send_goal(goal)
            self.move_client.wait_for_result(rospy.Duration(5.0))
            
            result = self.move_client.get_result()
            success = result and result.success
            final_width = self.max_gripper_width if success else 0.0
            message = "Opened (Franka)" if success else "Open failed"
        else:
            # MoveIt fallback
            joint_goal = self.gripper_group.get_current_joint_values()
            if len(joint_goal) >= 2:
                joint_goal[0] = 0.04
                joint_goal[1] = 0.04
                self.gripper_group.go(joint_goal, wait=True)
                self.gripper_group.stop()
            
            current_joints = self.gripper_group.get_current_joint_values()
            final_width = sum(current_joints) if len(current_joints) >= 2 else self.max_gripper_width
            success = True
            message = "Opened (MoveIt)"
        
        feedback.progress = 1.0
        feedback.status = "Open complete"
        feedback.current_width = final_width
        self.gripper_server.publish_feedback(feedback)
        
        return success, message, final_width
    
    def _grasp(self, target_width, force, feedback):
        """
        Grasp with force control (real) or position control (sim).
        Behavior tree decides what to do with the result.
        """
        feedback.status = f"Grasping (target: {target_width:.3f}m, force: {force}N)"
        feedback.progress = 0.0
        self.gripper_server.publish_feedback(feedback)
        
        # Force control (real robot)
        if self.use_force_control and self.grasp_client is not None:
            return self._grasp_force_control(target_width, force, feedback)
        
        # Position control (simulation)
        elif self.move_client is not None:
            return self._grasp_position_control(target_width, feedback)
        
        # MoveIt fallback
        else:
            return self._grasp_moveit_fallback(feedback)
    
    def _grasp_force_control(self, target_width, force, feedback):
        """Force-controlled grasp for real robot"""
        goal = GraspGoal()
        goal.width = target_width
        goal.epsilon.inner = self.grasp_epsilon_inner
        goal.epsilon.outer = self.grasp_epsilon_outer
        goal.speed = self.gripper_speed
        goal.force = force
        
        rospy.loginfo(f"Grasping (Force): {force}N @ {target_width}m")
        
        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result(rospy.Duration(10.0))
        
        result = self.grasp_client.get_result()
        current_joints = self.gripper_group.get_current_joint_values()
        final_width = sum(current_joints) if len(current_joints) >= 2 else 0.0
        
        if result and result.success:
            return True, f"Grasped (force={force}N, width={final_width:.4f}m)", final_width
        else:
            width_error = abs(final_width - target_width)
            
            # if final_width < 0.005:
            #     return False, f"No object (width={final_width:.4f}m)", final_width
            # elif width_error <= 0.02:
            #     return True, f"Acceptable (width={final_width:.4f}m)", final_width
            # else:
            #     return False, f"Mismatch: {final_width:.4f}m vs {target_width:.4f}m", final_width
            if final_width > 0.079: # Assuming 0.08 is max width
                return False, "Gripper remained open - likely missed", final_width
            else:
                return True, f"Grasped (force={force}N, width={final_width:.4f}m)", final_width
    
    def _grasp_position_control(self, target_width, feedback):
        """
        Position-controlled grasp for simulation.
        Enhanced for Gazebo stability [web:6][web:9].
        """
        sim_target = max(target_width, 0.001)
        
        goal = MoveGoal()
        goal.width = sim_target
        goal.speed = 0.05  # Slower for better contact detection
        
        rospy.loginfo(f"Grasping (Sim): {sim_target:.4f}m")
        self.move_client.send_goal(goal)
        
        # Monitor closure with stall detection
        timeout_time = rospy.Time.now() + rospy.Duration(3.0)
        last_width = 1.0
        stall_count = 0
        rate = rospy.Rate(10)  # 10 Hz monitoring
        
        while not rospy.is_shutdown() and rospy.Time.now() < timeout_time:
            # Check for preemption
            if self.gripper_server.is_preempt_requested():
                self.move_client.cancel_all_goals()
                return False, "Preempted", 0.0
            
            current_joints = self.gripper_group.get_current_joint_values()
            current_width = sum(current_joints) if len(current_joints) >= 2 else 0.0
            
            # Publish feedback
            progress = 0.3 + 0.5 * (1.0 - current_width / self.max_gripper_width)
            feedback.progress = min(progress, 0.9)
            feedback.status = f"Closing... {current_width:.4f}m"
            feedback.current_width = current_width
            self.gripper_server.publish_feedback(feedback)
            
            # Stall detection (fingers stopped)
            if abs(current_width - last_width) < 0.0001:
                stall_count += 1
                if stall_count >= 3:  # Stalled for 0.3s
                    break
            else:
                stall_count = 0
            
            last_width = current_width
            rate.sleep()
        
        # Let physics settle
        rospy.sleep(0.3)
        
        final_joints = self.gripper_group.get_current_joint_values()
        final_width = sum(final_joints) if len(final_joints) >= 2 else 0.0
        
        # Evaluate result - behavior tree decides next action
        if final_width < 0.003:
            self.move_client.cancel_all_goals()
            return False, f"No object (width={final_width:.4f}m)", final_width
        else:
            # Keep goal active to maintain pressure in Gazebo
            return True, f"Holding @ {final_width:.4f}m", final_width
    
    def _grasp_moveit_fallback(self, feedback):
        """MoveIt position control fallback"""
        rospy.logwarn("Using MoveIt fallback (no force control)")
        
        joint_goal = self.gripper_group.get_current_joint_values()
        if len(joint_goal) >= 2:
            joint_goal[0] = 0.001
            joint_goal[1] = 0.001
            self.gripper_group.go(joint_goal, wait=True)
        
        current_joints = self.gripper_group.get_current_joint_values()
        final_width = sum(current_joints) if len(current_joints) >= 2 else 0.0
        
        if final_width < 0.002:
            return False, "No object (MoveIt)", final_width
        else:
            return True, f"Closed (MoveIt, {final_width:.4f}m)", final_width
    
    def shutdown(self):
        """Clean shutdown"""
        rospy.loginfo("Shutting down control action server")
        self.move_group.stop()
        moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    try:
        server = ControlActionServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    finally:
        if 'server' in locals():
            server.shutdown()


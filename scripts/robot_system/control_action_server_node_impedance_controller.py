#!/usr/bin/env python3
import rospy
import moveit_commander
import actionlib
import tf
from franka_zed_gazebo.msg import (
    ControlActionAction, ControlActionFeedback, ControlActionResult,
    GripperActionAction, GripperActionFeedback, GripperActionResult
)
from sensor_msgs.msg import JointState

try:
    from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
    HAS_FRANKA_GRIPPER = True
except ImportError:
    HAS_FRANKA_GRIPPER = False

PANDA_INIT_JOINT_POSITIONS = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
PANDA_JOINT_NAMES = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]

class ControlActionServer:
    def __init__(self):
        rospy.init_node('control_action_server')
        moveit_commander.roscpp_initialize([])
        
        self.move_group = moveit_commander.MoveGroupCommander('panda_manipulator')
        self.gripper_group = moveit_commander.MoveGroupCommander('panda_hand')
        
        self.joint_pub = rospy.Publisher('/joint_impedance_controller/joint_command', JointState, queue_size=1)
        
        # Gripper parameters
        self.default_grasp_force  = rospy.get_param('~default_grasp_force', 20.0)
        self.grasp_epsilon_inner  = rospy.get_param('~grasp_epsilon_inner', 0.02)
        self.grasp_epsilon_outer  = rospy.get_param('~grasp_epsilon_outer', 0.02)
        self.gripper_speed        = rospy.get_param('~gripper_speed', 0.1)
        self.max_gripper_width    = rospy.get_param('~max_gripper_width', 0.08)
        self.use_force_control    = rospy.get_param('~use_force_control', True)

        # Initialize Franka Gripper clients
        self._init_gripper_clients()

        self._initializing = True
        rospy.sleep(1.0)
        self._move_to_init_pose()
        self._initializing = False

        # Action Servers
        self.control_server = actionlib.SimpleActionServer('/control_action', ControlActionAction, self.execute_control_cb, False)
        self.gripper_server = actionlib.SimpleActionServer('/gripper_action', GripperActionAction, self.execute_gripper_cb, False)
        
        self.control_server.start()
        self.gripper_server.start()
        rospy.loginfo("✔ Control and Gripper Action Servers Started")

    # ------------------------------------------------------------------
    def _init_gripper_clients(self):
        self.grasp_client = None
        self.move_client  = None

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
                rospy.loginfo("✔ Franka Gripper servers connected")
            else:
                rospy.logwarn("⚠ Franka Gripper unavailable. Using MoveIt.")
                self.grasp_client = None
                self.move_client  = None
        except Exception as e:
            rospy.logwarn(f"⚠ Gripper connection failed: {e}")
            self.grasp_client = None
            self.move_client  = None

    # --- TRAJECTORY LOGIC ---

    def _get_interpolation_points(self, points, elapsed):
        """Finds the two waypoints surrounding the current elapsed time."""
        for i in range(len(points) - 1):
            t0 = points[i].time_from_start.to_sec()
            t1 = points[i+1].time_from_start.to_sec()
            if t0 <= elapsed <= t1:
                return points[i], points[i+1], t0, t1
        return points[-1], points[-1], points[-1].time_from_start.to_sec(), points[-1].time_from_start.to_sec()

    def _move_to_init_pose(self):
        rospy.loginfo("Moving to init joint pose...")
        self.move_group.set_joint_value_target(PANDA_INIT_JOINT_POSITIONS)
        success, plan, _, _ = self.move_group.plan()
        if success:
            self._execute_with_monitoring(plan, ControlActionFeedback())

    def execute_control_cb(self, goal):
        success = self._execute_with_monitoring(goal.trajectory, ControlActionFeedback())
        result = ControlActionResult(success=success)
        if success:
            self.control_server.set_succeeded(result)
        else:
            self.control_server.set_aborted(result)

    def _execute_with_monitoring(self, trajectory, feedback, rate_hz=200):
        traj = trajectory.joint_trajectory
        points = traj.points
        names = list(traj.joint_names)
        
        if not points: return False

        # Continuous start: interpolation begins from actual current position
        current_pos = self.move_group.get_current_joint_values()
        total_duration = points[-1].time_from_start.to_sec()
        rate = rospy.Rate(rate_hz)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            if not self._initializing and self.control_server.is_preempt_requested():
                return False

            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed >= total_duration:
                self._publish_cmd(names, points[-1].positions)
                break

            p0, p1, t0, t1 = self._get_interpolation_points(points, elapsed)
            
            # Use current real position for the very first segment to avoid 'jumps'
            start_vals = current_pos if elapsed < points[0].time_from_start.to_sec() else p0.positions
            
            alpha = (elapsed - t0) / (t1 - t0) if (t1 - t0) > 1e-6 else 1.0
            interp = [start_vals[j] + alpha * (p1.positions[j] - start_vals[j]) for j in range(len(names))]
            
            self._publish_cmd(names, interp)
            rate.sleep()
        return True

    def _publish_cmd(self, names, positions):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = names
        msg.position = list(positions)
        msg.velocity = [0.0] * len(positions)
        self.joint_pub.publish(msg)

    # ========== GRIPPER ACTION CALLBACKS ==========

    def gripper_preempt_cb(self):
        rospy.logwarn("Gripper action preempted by behavior tree")
        if self.move_client:
            self.move_client.cancel_all_goals()
        if self.grasp_client:
            self.grasp_client.cancel_all_goals()

    def execute_gripper_cb(self, goal):
        feedback = GripperActionFeedback()
        result   = GripperActionResult()

        try:
            if goal.open:
                success, message, final_width = self._open_gripper(feedback)
            else:
                target_width = goal.width if goal.width > 0 else 0.0
                grasp_force  = goal.force if goal.force > 0 else self.default_grasp_force
                success, message, final_width = self._grasp(target_width, grasp_force, feedback)

            result.success     = success
            result.message     = message
            result.final_width = final_width

            if success:
                rospy.loginfo(f"✔ {message}")
                self.gripper_server.set_succeeded(result)
            else:
                rospy.logwarn(f"✗ {message}")
                self.gripper_server.set_aborted(result)

        except Exception as e:
            result.success     = False
            result.message     = f"Error: {str(e)}"
            result.final_width = -1.0
            rospy.logerr(result.message)
            self.gripper_server.set_aborted(result)

    # ========== GRIPPER CONTROL METHODS ==========

    def _open_gripper(self, feedback):
        feedback.status   = "Opening gripper"
        feedback.progress = 0.0
        self.gripper_server.publish_feedback(feedback)

        if self.move_client is not None:
            goal       = MoveGoal()
            goal.width = self.max_gripper_width
            goal.speed = self.gripper_speed
            self.move_client.send_goal(goal)
            self.move_client.wait_for_result(rospy.Duration(5.0))
            result      = self.move_client.get_result()
            success     = result and result.success
            final_width = self.max_gripper_width if success else 0.0
            message     = "Opened (Franka)" if success else "Open failed"
        else:
            joint_goal = self.gripper_group.get_current_joint_values()
            if len(joint_goal) >= 2:
                joint_goal[0] = 0.04
                joint_goal[1] = 0.04
                self.gripper_group.go(joint_goal, wait=True)
                self.gripper_group.stop()
            current_joints = self.gripper_group.get_current_joint_values()
            final_width    = sum(current_joints) if len(current_joints) >= 2 else self.max_gripper_width
            success        = True
            message        = "Opened (MoveIt)"

        feedback.progress     = 1.0
        feedback.status       = "Open complete"
        feedback.current_width = final_width
        self.gripper_server.publish_feedback(feedback)
        return success, message, final_width

    def _grasp(self, target_width, force, feedback):
        feedback.status   = f"Grasping (target: {target_width:.3f}m, force: {force}N)"
        feedback.progress = 0.0
        self.gripper_server.publish_feedback(feedback)

        if self.use_force_control and self.grasp_client is not None:
            return self._grasp_force_control(target_width, force, feedback)
        elif self.move_client is not None:
            return self._grasp_position_control(target_width, feedback)
        else:
            return self._grasp_moveit_fallback(feedback)

    def _grasp_force_control(self, target_width, force, feedback):
        goal               = GraspGoal()
        goal.width         = target_width
        goal.epsilon.inner = self.grasp_epsilon_inner
        goal.epsilon.outer = self.grasp_epsilon_outer
        goal.speed         = self.gripper_speed
        goal.force         = force
        rospy.loginfo(f"Grasping (Force): {force}N @ {target_width}m")
        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result(rospy.Duration(10.0))
        result         = self.grasp_client.get_result()
        current_joints = self.gripper_group.get_current_joint_values()
        final_width    = sum(current_joints) if len(current_joints) >= 2 else 0.0
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

            # Evaluate result - behavior tree decides next action
            if final_width < 0.003:
                self.move_client.cancel_all_goals()
                return False, f"No object (width={final_width:.4f}m)", final_width
            else:
                # Keep goal active to maintain pressure in Gazebo
                return True, f"Holding @ {final_width:.4f}m", final_width


    def _grasp_position_control(self, target_width, feedback):
        sim_target    = max(target_width, 0.001)
        goal          = MoveGoal()
        goal.width    = sim_target
        goal.speed    = 0.05
        rospy.loginfo(f"Grasping (Sim): {sim_target:.4f}m")
        self.move_client.send_goal(goal)
        timeout_time = rospy.Time.now() + rospy.Duration(3.0)
        last_width   = 1.0
        stall_count  = 0
        rate         = rospy.Rate(10)
        while not rospy.is_shutdown() and rospy.Time.now() < timeout_time:
            if self.gripper_server.is_preempt_requested():
                self.move_client.cancel_all_goals()
                return False, "Preempted", 0.0
            current_joints = self.gripper_group.get_current_joint_values()
            current_width  = sum(current_joints) if len(current_joints) >= 2 else 0.0
            progress       = 0.3 + 0.5 * (1.0 - current_width / self.max_gripper_width)
            feedback.progress     = min(progress, 0.9)
            feedback.status       = f"Closing... {current_width:.4f}m"
            feedback.current_width = current_width
            self.gripper_server.publish_feedback(feedback)
            if abs(current_width - last_width) < 0.0001:
                stall_count += 1
                if stall_count >= 3:
                    break
            else:
                stall_count = 0
            last_width = current_width
            rate.sleep()
        rospy.sleep(0.3)
        final_joints = self.gripper_group.get_current_joint_values()
        final_width  = sum(final_joints) if len(final_joints) >= 2 else 0.0
        if final_width < 0.003:
            self.move_client.cancel_all_goals()
            return False, f"No object (width={final_width:.4f}m)", final_width
        else:
            return True, f"Holding @ {final_width:.4f}m", final_width

    def _grasp_moveit_fallback(self, feedback):
        rospy.logwarn("Using MoveIt fallback (no force control)")
        joint_goal = self.gripper_group.get_current_joint_values()
        if len(joint_goal) >= 2:
            joint_goal[0] = 0.001
            joint_goal[1] = 0.001
            self.gripper_group.go(joint_goal, wait=True)
        current_joints = self.gripper_group.get_current_joint_values()
        final_width    = sum(current_joints) if len(current_joints) >= 2 else 0.0
        if final_width < 0.002:
            return False, "No object (MoveIt)", final_width
        else:
            return True, f"Closed (MoveIt, {final_width:.4f}m)", final_width

    def shutdown(self):
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
#!/usr/bin/env python3
"""
control_action_server_node_impedance_controller.py

Fixes applied vs original:
  1. Always populate JointState.name so the C++ controller can reorder by name.
  2. Added move_to_init_pose() called once at startup so the robot goes to a
     known safe configuration before accepting goals.
  3. Trajectory joint name order is now explicitly preserved end-to-end.
"""

import rospy
import moveit_commander
import actionlib

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
    rospy.logwarn("franka_gripper not found. Using MoveIt fallback.")


# Panda ready/home pose in joint space (radians) — adjust to your setup
PANDA_INIT_JOINT_POSITIONS = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]

# Canonical joint order that matches your YAML / URDF
PANDA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
]


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
        planning_group    = rospy.get_param('~planning_group', 'panda_manipulator')
        gripper_group_name = rospy.get_param('~gripper_group', 'panda_hand')

        self.move_group = moveit_commander.MoveGroupCommander(
            planning_group, wait_for_servers=30
        )
        self.gripper_group = moveit_commander.MoveGroupCommander(
            gripper_group_name, wait_for_servers=30
        )

        # Publisher for impedance controller
        self.joint_pub = rospy.Publisher(
            '/joint_impedance_controller/joint_command',
            JointState,
            queue_size=1
        )

        # Motion parameters
        self.max_velocity    = rospy.get_param('~max_velocity', 0.3)   # conservative on real robot
        self.max_acceleration = rospy.get_param('~max_acceleration', 0.3)
        self.move_group.set_max_velocity_scaling_factor(self.max_velocity)
        self.move_group.set_max_acceleration_scaling_factor(self.max_acceleration)

        # Gripper parameters
        self.default_grasp_force  = rospy.get_param('~default_grasp_force', 20.0)
        self.grasp_epsilon_inner  = rospy.get_param('~grasp_epsilon_inner', 0.02)
        self.grasp_epsilon_outer  = rospy.get_param('~grasp_epsilon_outer', 0.02)
        self.gripper_speed        = rospy.get_param('~gripper_speed', 0.1)
        self.max_gripper_width    = rospy.get_param('~max_gripper_width', 0.08)
        self.use_force_control    = rospy.get_param('~use_force_control', True)

        # Initialize Franka Gripper clients
        self._init_gripper_clients()

        # Flag used by _execute_with_monitoring to skip preempt check
        # during init (control_server doesn't exist yet at this point)
        self._initializing = True

        # ----------------------------------------------------------------
        # FIX 2: Move to a known init pose so the robot is never in a
        # random configuration when the impedance controller starts.
        # This is streamed over the joint_pub topic, so the impedance
        # controller must already be running before this node starts.
        # ----------------------------------------------------------------
        rospy.sleep(0.5)   # give controller a moment to fully start
        self._move_to_init_pose()
        self._initializing = False

        # Create action servers
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

        rospy.loginfo(f"✔ Control Action Server: {planning_group} "
                      f"(vel={self.max_velocity}, acc={self.max_acceleration})")
        rospy.loginfo(f"✔ Gripper Action Server: {gripper_group_name} "
                      f"(force_control={self.use_force_control and self.grasp_client is not None})")

    # ------------------------------------------------------------------
    # FIX 2: Init pose via MoveIt plan → stream to impedance controller
    # ------------------------------------------------------------------
    def _move_to_init_pose(self):
        """
        Plan to the Panda home/ready pose using MoveIt and stream it to
        the impedance controller.  Uses the same streaming path as normal
        trajectory execution so there is no mode-switch jerk.
        """
        init_positions = rospy.get_param(
            '~init_joint_positions', PANDA_INIT_JOINT_POSITIONS
        )
        rospy.loginfo("Moving to init joint pose via impedance controller …")

        self.move_group.set_joint_value_target(init_positions)
        plan = self.move_group.plan()

        # plan() returns (success, trajectory, planning_time, error_code) in newer MoveIt
        if isinstance(plan, tuple):
            success, trajectory, _, _ = plan
        else:
            trajectory = plan
            success = (len(trajectory.joint_trajectory.points) > 0)

        if not success or len(trajectory.joint_trajectory.points) == 0:
            rospy.logwarn("Init pose planning failed — holding current position.")
            return

        dummy_feedback = ControlActionFeedback()
        self._execute_with_monitoring(trajectory, dummy_feedback)
        rospy.loginfo("✔ Init pose reached.")

    # ------------------------------------------------------------------

    def _init_gripper_clients(self):
        self.grasp_client = None
        self.move_client  = None

        if not HAS_FRANKA_GRIPPER:
            return

        try:
            self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
            self.move_client  = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)

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

    # ========== CONTROL ACTION CALLBACKS ==========

    def control_preempt_cb(self):
        rospy.logwarn("Control action preempted by behavior tree")
        self.move_group.stop()

    def execute_control_cb(self, goal):
        feedback = ControlActionFeedback()
        result   = ControlActionResult()

        try:
            num_points = len(goal.trajectory.joint_trajectory.points)
            if num_points == 0:
                result.success = False
                result.message = "Empty trajectory"
                self.control_server.set_aborted(result)
                return

            rospy.loginfo(f"▶ Executing trajectory: {num_points} waypoints")

            feedback.progress = 0.0
            feedback.status   = "Starting execution"
            self.control_server.publish_feedback(feedback)

            success = self._execute_with_monitoring(goal.trajectory, feedback)

            feedback.progress = 1.0
            feedback.status   = "Complete" if success else "Failed"
            self.control_server.publish_feedback(feedback)

            result.success = success
            result.message = "Executed" if success else "Failed"

            if success:
                rospy.loginfo(f"✔ {result.message}")
                self.control_server.set_succeeded(result)
            else:
                rospy.logwarn(f"✗ {result.message}")
                self.control_server.set_aborted(result)

        except Exception as e:
            result.success = False
            result.message = f"Error: {str(e)}"
            rospy.logerr(result.message)
            self.move_group.stop()
            self.control_server.set_aborted(result)

    def _execute_with_monitoring(self, trajectory, feedback, rate_hz=200):
        """
        Stream trajectory points to the impedance controller.

        FIX 1: JointState.name is ALWAYS populated so the C++ controller
        can reorder joints by name rather than relying on positional order.
        """
        traj        = trajectory.joint_trajectory
        points      = traj.points
        joint_names = list(traj.joint_names)   # preserve MoveIt ordering

        if not points:
            return False

        total_duration = points[-1].time_from_start.to_sec()
        publish_rate   = rospy.Rate(rate_hz)
        start_time     = rospy.Time.now()

        while not rospy.is_shutdown():
            if not self._initializing and self.control_server.is_preempt_requested():
                return False

            elapsed = (rospy.Time.now() - start_time).to_sec()

            if elapsed >= total_duration:
                # Hold final point
                p   = points[-1]
                cmd = JointState()
                cmd.header.stamp = rospy.Time.now()
                cmd.name         = joint_names          # FIX 1 — always set name
                cmd.position     = list(p.positions)
                cmd.velocity     = [0.0] * len(joint_names)
                self.joint_pub.publish(cmd)
                break

            # Find surrounding waypoints for interpolation
            t0, t1, p0, p1 = None, None, None, None
            for k in range(len(points) - 1):
                a = points[k].time_from_start.to_sec()
                b = points[k + 1].time_from_start.to_sec()
                if a <= elapsed <= b:
                    t0, t1, p0, p1 = a, b, points[k], points[k + 1]
                    break

            if p0 is None:
                # elapsed < first waypoint time — hold first point
                p0 = p1 = points[0]
                t0 = t1 = 0.0

            alpha_interp = (elapsed - t0) / (t1 - t0) if (t1 - t0) > 1e-6 else 1.0
            alpha_interp = max(0.0, min(1.0, alpha_interp))

            interp_pos = [
                p0.positions[j] + alpha_interp * (p1.positions[j] - p0.positions[j])
                for j in range(len(joint_names))
            ]

            if p0.velocities and p1.velocities and len(p0.velocities) >= len(joint_names):
                interp_vel = [
                    p0.velocities[j] + alpha_interp * (p1.velocities[j] - p0.velocities[j])
                    for j in range(len(joint_names))
                ]
            else:
                interp_vel = [0.0] * len(joint_names)

            cmd = JointState()
            cmd.header.stamp = rospy.Time.now()
            cmd.name         = joint_names   # FIX 1 — always set name
            cmd.position     = interp_pos
            cmd.velocity     = interp_vel
            self.joint_pub.publish(cmd)

            feedback.progress = elapsed / total_duration
            feedback.status   = f"Streaming: {int(feedback.progress * 100)}%"
            publish_rate.sleep()

        return True

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
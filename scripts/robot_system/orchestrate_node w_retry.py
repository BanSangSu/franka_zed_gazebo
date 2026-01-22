#!/usr/bin/env python3
"""
Advanced Orchestrator Node - Reactive Behavior Tree for Manipulation
Compatible with py_trees 0.7.6 (ROS Noetic)
"""

import rospy
import py_trees
import py_trees_ros
import actionlib
from copy import deepcopy
from enum import Enum
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Vector3
from franka_zed_gazebo.srv import PerceptionService
from franka_zed_gazebo.msg import (
    ControlActionAction, ControlActionGoal,
    GripperActionAction, GripperActionGoal,
    PlanningActionAction, PlanningActionGoal
)


###############################################################################
# CONFIGURATION
###############################################################################

class RobotConfig:
    """Robot workspace and manipulation parameters"""
    # Workspace
    BASE_POSITION = (0.5, 0.0, 0.0225)
    CUBE_SIZE = 0.045
    
    # Approach heights
    SAFE_HEIGHT = 0.15
    GRASP_HEIGHT = 0.01
    PLACE_HEIGHT = 0.01
    
    # Gripper
    GRIPPER_OPEN_WIDTH = 0.08
    GRIPPER_FORCE = 30.0
    GRIPPER_WIDTH_MARGIN = 0.005
    MIN_GRIPPER_WIDTH = 0.01
    
    # Orientation (top-down grasp)
    GRASP_ORIENTATION = [1.0, 0.0, 0.0, 0.0]
    
    # Retry parameters
    MAX_GRASP_RETRIES = 2
    MAX_PLAN_RETRIES = 3


class TaskPattern(Enum):
    """Supported manipulation patterns"""
    STACK = "STACK"
    PYRAMID = "PYRAMID"
    LINE = "LINE"


###############################################################################
# BLACKBOARD KEYS
###############################################################################

class BB:
    """Blackboard key namespace"""
    CUBES = "cubes"
    GOALS = "goals"
    GOAL_INDEX = "goal_idx"
    COMPLETED_GOALS = "completed"
    SELECTED_CUBE = "selected_cube"
    TARGET_POSE = "target_pose"
    PLANNED_TRAJECTORY = "trajectory"
    ALLOWED_COLLISION = "allowed_collision"


def init_blackboard():
    """Initialize blackboard with default values"""
    blackboard = py_trees.blackboard.Blackboard()
    blackboard.set(BB.CUBES, [])
    blackboard.set(BB.GOALS, [])
    blackboard.set(BB.GOAL_INDEX, 0)
    blackboard.set(BB.COMPLETED_GOALS, set())
    blackboard.set(BB.PLANNED_TRAJECTORY, None)
    blackboard.set(BB.ALLOWED_COLLISION, "")
    return blackboard


###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def quaternion_multiply(q1, q2):
    """Multiply two quaternions [x,y,z,w]"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]


def create_pose(x, y, z, quat=None):
    """Create Pose message"""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    if quat is None:
        pose.orientation.w = 1.0
    else:
        pose.orientation.x, pose.orientation.y = quat[0], quat[1]
        pose.orientation.z, pose.orientation.w = quat[2], quat[3]
    return pose


def offset_pose(base_pose, dx=0, dy=0, dz=0, orientation=None):
    """Create offset pose with optional orientation override"""
    pose = deepcopy(base_pose)
    pose.position.x += dx
    pose.position.y += dy
    pose.position.z += dz
    
    if orientation is not None:
        orig_q = [base_pose.orientation.x, base_pose.orientation.y,
                 base_pose.orientation.z, base_pose.orientation.w]
        new_q = quaternion_multiply(orig_q, orientation)
        pose.orientation.x, pose.orientation.y = new_q[0], new_q[1]
        pose.orientation.z, pose.orientation.w = new_q[2], new_q[3]
    
    return pose


class RetryDecorator(py_trees.decorators.Decorator):
    """
    Custom Retry decorator for py_trees 0.7.6
    Retries child behavior on failure up to num_failures times
    """
    
    def __init__(self, name, child, num_failures=3):
        super(RetryDecorator, self).__init__(name=name, child=child)
        self.num_failures = num_failures
        self.failure_count = 0
    
    def initialise(self):
        """Reset failure counter"""
        self.failure_count = 0
    
    def update(self):
        """Retry logic"""
        if self.decorated.status == py_trees.common.Status.FAILURE:
            self.failure_count += 1
            
            if self.failure_count < self.num_failures:
                # Reset child and retry
                self.decorated.stop(py_trees.common.Status.INVALID)
                self.decorated.setup(timeout=15)
                self.feedback_message = f"Retry {self.failure_count}/{self.num_failures}"
                rospy.logwarn(f"[{self.name}] {self.feedback_message}")
                return py_trees.common.Status.RUNNING
            else:
                # Max retries reached
                self.feedback_message = f"Failed after {self.num_failures} attempts"
                rospy.logerr(f"[{self.name}] {self.feedback_message}")
                return py_trees.common.Status.FAILURE
        
        elif self.decorated.status == py_trees.common.Status.RUNNING:
            return py_trees.common.Status.RUNNING
        
        else:  # SUCCESS
            self.feedback_message = "Success"
            return py_trees.common.Status.SUCCESS

###############################################################################
# PATTERN GENERATORS
###############################################################################

class PatternGenerator:
    """Generate goal poses for different patterns"""
    
    @staticmethod
    def generate(pattern_type, value):
        """Generate list of goal poses"""
        generators = {
            TaskPattern.STACK: PatternGenerator._stack,
            TaskPattern.PYRAMID: PatternGenerator._pyramid,
            TaskPattern.LINE: PatternGenerator._line,
        }
        return generators[pattern_type](value)
    
    @staticmethod
    def _stack(num_cubes):
        """Vertical stack of N cubes"""
        goals = []
        x, y, z = RobotConfig.BASE_POSITION
        for i in range(int(num_cubes)):
            pose = create_pose(x, y, z + i * RobotConfig.CUBE_SIZE)
            goals.append(pose)
        return goals
    
    @staticmethod
    def _pyramid(num_cubes):
        """3-2-1 pyramid structure"""
        goals = []
        x, y, z = RobotConfig.BASE_POSITION
        cube_size = RobotConfig.CUBE_SIZE
        
        # Layer 1: 3 cubes
        for y_offset in [-cube_size, 0, cube_size]:
            goals.append(create_pose(x, y + y_offset, z))
        
        # Layer 2: 2 cubes
        for y_offset in [-cube_size/2, cube_size/2]:
            goals.append(create_pose(x, y + y_offset, z + cube_size))
        
        # Layer 3: 1 cube
        goals.append(create_pose(x, y, z + 2*cube_size))
        return goals
    
    @staticmethod
    def _line(num_cubes):
        """Horizontal line of cubes"""
        goals = []
        x, y, z = RobotConfig.BASE_POSITION
        cube_size = RobotConfig.CUBE_SIZE
        start_y = y - (int(num_cubes) - 1) * cube_size / 2
        for i in range(int(num_cubes)):
            goals.append(create_pose(x, start_y + i * cube_size, z))
        return goals


###############################################################################
# BASE BEHAVIOR CLASSES
###############################################################################

class ServiceBehaviour(py_trees.behaviour.Behaviour):
    """Base class for ROS service calls"""
    
    def __init__(self, name, service_name, service_type):
        super(ServiceBehaviour, self).__init__(name)
        self.service_name = service_name
        self.service_type = service_type
        self.proxy = None
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def setup(self, timeout):
        try:
            rospy.wait_for_service(self.service_name, timeout=timeout)
            self.proxy = rospy.ServiceProxy(self.service_name, self.service_type)
            rospy.loginfo(f"[{self.name}] Connected to {self.service_name}")
            return True
        except rospy.ROSException as e:
            rospy.logerr(f"[{self.name}] Service unavailable: {e}")
            return False


class ActionBehaviour(py_trees.behaviour.Behaviour):
    """Base class for action client behaviors"""
    
    def __init__(self, name, action_name, action_type):
        super(ActionBehaviour, self).__init__(name)
        self.action_name = action_name
        self.action_type = action_type
        self.client = None
        self.goal_active = False
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def setup(self, timeout):
        self.client = actionlib.SimpleActionClient(self.action_name, self.action_type)
        if self.client.wait_for_server(rospy.Duration(timeout)):
            rospy.loginfo(f"[{self.name}] Connected to {self.action_name}")
            return True
        rospy.logerr(f"[{self.name}] Action server unavailable")
        return False
    
    def initialise(self):
        self.goal_active = False
    
    def update(self):
        if not self.goal_active:
            goal = self.construct_goal()
            if goal is None:
                return py_trees.common.Status.FAILURE
            self.client.send_goal(goal)
            self.goal_active = True
            return py_trees.common.Status.RUNNING
        
        state = self.client.get_state()
        
        if state == GoalStatus.SUCCEEDED:
            return self.handle_success(self.client.get_result())
        elif state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
            return self.handle_failure()
        
        return py_trees.common.Status.RUNNING
    
    def construct_goal(self):
        """Override to construct specific goal"""
        raise NotImplementedError
    
    def handle_success(self, result):
        """Override to handle result"""
        return py_trees.common.Status.SUCCESS
    
    def handle_failure(self):
        """Override to handle failure"""
        return py_trees.common.Status.FAILURE


###############################################################################
# PERCEPTION
###############################################################################

class PerceiveCubes(ServiceBehaviour):
    """Detect cubes in workspace"""
    
    def __init__(self):
        super(PerceiveCubes, self).__init__("Perceive", "/perception_service", PerceptionService)
    
    def update(self):
        try:
            response = self.proxy(trigger=True)
            
            if not response.success:
                rospy.logwarn(f"[{self.name}] Perception failed: {response.message}")
                return py_trees.common.Status.FAILURE
            
            cubes = []
            for i in range(response.num_cubes):
                cubes.append({
                    'id': i,
                    'pose': response.cube_poses.poses[i],
                    'dimensions': response.dimensions[i],
                    'label': response.labels[i] if i < len(response.labels) else "cube",
                    'confidence': response.confidences[i] if i < len(response.confidences) else 1.0
                })
            
            self.blackboard.set(BB.CUBES, cubes)
            rospy.loginfo(f"[{self.name}] Detected {len(cubes)} cubes")
            return py_trees.common.Status.SUCCESS
            
        except Exception as e:
            rospy.logerr(f"[{self.name}] Exception: {e}")
            return py_trees.common.Status.FAILURE


###############################################################################
# TASK MANAGEMENT
###############################################################################

class InitializeTask(py_trees.behaviour.Behaviour):
    """Initialize task goals"""
    
    def __init__(self, pattern_type, pattern_value):
        super(InitializeTask, self).__init__("InitTask")
        self.pattern_type = pattern_type
        self.pattern_value = pattern_value
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def update(self):
        if self.blackboard.get(BB.GOALS):
            return py_trees.common.Status.SUCCESS
        
        goals = PatternGenerator.generate(self.pattern_type, self.pattern_value)
        self.blackboard.set(BB.GOALS, goals)
        self.blackboard.set(BB.COMPLETED_GOALS, set())
        self.blackboard.set(BB.GOAL_INDEX, 0)
        
        rospy.loginfo(f"[{self.name}] Initialized {len(goals)} goals for {self.pattern_type.value}")
        return py_trees.common.Status.SUCCESS


class SelectNextGoal(py_trees.behaviour.Behaviour):
    """Select next incomplete goal"""
    
    def __init__(self):
        super(SelectNextGoal, self).__init__("SelectGoal")
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def update(self):
        goals = self.blackboard.get(BB.GOALS)
        completed = self.blackboard.get(BB.COMPLETED_GOALS)
        
        for idx, goal_pose in enumerate(goals):
            if idx not in completed:
                self.blackboard.set(BB.GOAL_INDEX, idx)
                self.blackboard.set(BB.TARGET_POSE, goal_pose)
                rospy.loginfo(f"[{self.name}] Goal {idx+1}/{len(goals)} selected")
                return py_trees.common.Status.SUCCESS
        
        rospy.loginfo(f"[{self.name}] All {len(goals)} goals completed!")
        return py_trees.common.Status.FAILURE


class SelectBestCube(py_trees.behaviour.Behaviour):
    """Select optimal cube for current goal"""
    
    def __init__(self):
        super(SelectBestCube, self).__init__("SelectCube")
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def update(self):
        cubes = self.blackboard.get(BB.CUBES)
        completed = self.blackboard.get(BB.COMPLETED_GOALS)
        
        # Filter available cubes (not already placed)
        available = [c for c in cubes if c['id'] not in completed]
        
        if not available:
            rospy.logwarn(f"[{self.name}] No available cubes")
            return py_trees.common.Status.FAILURE
        
        # Select closest cube to robot (largest X)
        best = max(available, key=lambda c: c['pose'].position.x)
        self.blackboard.set(BB.SELECTED_CUBE, best)
        
        rospy.loginfo(f"[{self.name}] Selected cube {best['id']}")
        return py_trees.common.Status.SUCCESS


class SetCollisionObject(py_trees.behaviour.Behaviour):
    """Determine allowed collision object for stacking"""
    
    def __init__(self):
        super(SetCollisionObject, self).__init__("SetCollision")
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def update(self):
        goal_idx = self.blackboard.get(BB.GOAL_INDEX)
        completed = self.blackboard.get(BB.COMPLETED_GOALS)
        
        collision_obj = ""
        
        # If stacking on previous goal, allow collision with it
        if goal_idx > 0 and (goal_idx - 1) in completed:
            prev_cube_id = goal_idx - 1
            collision_obj = f"placed_cube_{prev_cube_id}"
            rospy.loginfo(f"[{self.name}] Allowing collision with {collision_obj}")
        
        self.blackboard.set(BB.ALLOWED_COLLISION, collision_obj)
        return py_trees.common.Status.SUCCESS


class MarkGoalComplete(py_trees.behaviour.Behaviour):
    """Mark current goal as completed"""
    
    def __init__(self):
        super(MarkGoalComplete, self).__init__("MarkComplete")
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def update(self):
        goal_idx = self.blackboard.get(BB.GOAL_INDEX)
        completed = self.blackboard.get(BB.COMPLETED_GOALS)
        completed.add(goal_idx)
        self.blackboard.set(BB.COMPLETED_GOALS, completed)
        
        rospy.loginfo(f"[{self.name}] Goal {goal_idx} marked complete")
        return py_trees.common.Status.SUCCESS


###############################################################################
# MOTION PLANNING
###############################################################################

class PlanToHome(ActionBehaviour):
    """Plan motion to home position"""
    
    def __init__(self):
        super(PlanToHome, self).__init__("PlanHome", "/planning_action", PlanningActionAction)
    
    def construct_goal(self):
        goal = PlanningActionGoal()
        goal.action = "HOME"
        return goal
    
    def handle_success(self, result):
        if result and result.success:
            self.blackboard.set(BB.PLANNED_TRAJECTORY, result.trajectory)
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class PlanToPose(ActionBehaviour):
    """Plan motion to specified pose"""
    
    def __init__(self, name, pose_source, offset=(0,0,0), use_collision=False):
        super(PlanToPose, self).__init__(name, "/planning_action", PlanningActionAction)
        self.pose_source = pose_source
        self.offset = offset
        self.use_collision = use_collision
    
    def construct_goal(self):
        source = self.blackboard.get(self.pose_source)
        
        if self.pose_source == BB.SELECTED_CUBE:
            base_pose = source['pose']
            goal_msg = PlanningActionGoal()
            goal_msg.allowed_collision_object = f"cube_{source['id']}"
        else:
            base_pose = source
            goal_msg = PlanningActionGoal()
            if self.use_collision:
                goal_msg.allowed_collision_object = self.blackboard.get(BB.ALLOWED_COLLISION)
        
        target = offset_pose(base_pose, *self.offset, 
                           orientation=RobotConfig.GRASP_ORIENTATION)
        goal_msg.target_pose = target
        goal_msg.action = ""
        
        return goal_msg
    
    def handle_success(self, result):
        if result and result.success:
            self.blackboard.set(BB.PLANNED_TRAJECTORY, result.trajectory)
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class AttachObject(ActionBehaviour):
    """Attach cube to gripper"""
    
    def __init__(self):
        super(AttachObject, self).__init__("Attach", "/planning_action", PlanningActionAction)
    
    def construct_goal(self):
        cube = self.blackboard.get(BB.SELECTED_CUBE)
        goal = PlanningActionGoal()
        goal.action = "ATTACH"
        goal.object_name = f"cube_{cube['id']}"
        return goal
    
    def handle_success(self, result):
        return py_trees.common.Status.SUCCESS if result.success else py_trees.common.Status.FAILURE


class DetachObject(ActionBehaviour):
    """Detach cube from gripper"""
    
    def __init__(self):
        super(DetachObject, self).__init__("Detach", "/planning_action", PlanningActionAction)
    
    def construct_goal(self):
        cube = self.blackboard.get(BB.SELECTED_CUBE)
        goal = PlanningActionGoal()
        goal.action = "DETACH"
        goal.object_name = f"cube_{cube['id']}"
        return goal


###############################################################################
# MOTION EXECUTION
###############################################################################

class ExecuteTrajectory(ActionBehaviour):
    """Execute planned trajectory"""
    
    def __init__(self):
        super(ExecuteTrajectory, self).__init__("Execute", "/control_action", ControlActionAction)
    
    def construct_goal(self):
        trajectory = self.blackboard.get(BB.PLANNED_TRAJECTORY)
        if trajectory is None:
            rospy.logerr(f"[{self.name}] No trajectory available")
            return None
        
        goal = ControlActionGoal()
        goal.trajectory = trajectory
        return goal


###############################################################################
# GRIPPER CONTROL
###############################################################################

class OpenGripper(ActionBehaviour):
    """Open gripper"""
    
    def __init__(self):
        super(OpenGripper, self).__init__("Open", "/gripper_action", GripperActionAction)
    
    def construct_goal(self):
        goal = GripperActionGoal()
        goal.open = True
        goal.width = RobotConfig.GRIPPER_OPEN_WIDTH
        return goal
    
    def handle_success(self, result):
        return py_trees.common.Status.SUCCESS


class CloseGripper(ActionBehaviour):
    """Close gripper to grasp"""
    
    def __init__(self):
        super(CloseGripper, self).__init__("Close", "/gripper_action", GripperActionAction)
    
    def construct_goal(self):
        cube = self.blackboard.get(BB.SELECTED_CUBE)
        dims = cube['dimensions']
        
        dx = dims.x if hasattr(dims, 'x') else dims[0]
        dy = dims.y if hasattr(dims, 'y') else dims[1]
        
        goal = GripperActionGoal()
        goal.open = False
        goal.width = max(RobotConfig.MIN_GRIPPER_WIDTH, 
                        min(dx, dy) - RobotConfig.GRIPPER_WIDTH_MARGIN)
        goal.force = RobotConfig.GRIPPER_FORCE
        return goal
    
    def handle_success(self, result):
        if result and result.success:
            rospy.loginfo(f"[{self.name}] Grasp successful")
            return py_trees.common.Status.SUCCESS
        else:
            rospy.logwarn(f"[{self.name}] Grasp failed - object slipped")
            return py_trees.common.Status.FAILURE


###############################################################################
# BEHAVIOR TREE CONSTRUCTION
###############################################################################

def create_pick_sequence():
    """Create pick behavior subtree"""
    pick = py_trees.composites.Sequence("PickCube")
    
    # Approach
    pick.add_child(OpenGripper())
    pick.add_child(PlanToPose("PlanApproach", BB.SELECTED_CUBE, 
                              offset=(0, 0, RobotConfig.SAFE_HEIGHT)))
    pick.add_child(ExecuteTrajectory())
    
    # Descend
    pick.add_child(PlanToPose("PlanDescend", BB.SELECTED_CUBE,
                             offset=(0, 0, RobotConfig.GRASP_HEIGHT)))
    pick.add_child(ExecuteTrajectory())
    
    # Grasp
    pick.add_child(CloseGripper())
    pick.add_child(AttachObject())
    
    # Lift
    pick.add_child(PlanToPose("PlanLift", BB.SELECTED_CUBE,
                             offset=(0, 0, RobotConfig.SAFE_HEIGHT)))
    pick.add_child(ExecuteTrajectory())
    
    return pick


def create_place_sequence():
    """Create place behavior subtree"""
    place = py_trees.composites.Sequence("PlaceCube")
    
    # Approach goal
    place.add_child(SetCollisionObject())
    place.add_child(PlanToPose("PlanApproachGoal", BB.TARGET_POSE,
                              offset=(0, 0, RobotConfig.SAFE_HEIGHT),
                              use_collision=True))
    place.add_child(ExecuteTrajectory())
    
    # Descend to place
    place.add_child(PlanToPose("PlanDescendGoal", BB.TARGET_POSE,
                              offset=(0, 0, RobotConfig.PLACE_HEIGHT),
                              use_collision=True))
    place.add_child(ExecuteTrajectory())
    
    # Release
    place.add_child(OpenGripper())
    place.add_child(DetachObject())
    place.add_child(MarkGoalComplete())
    
    # Clear
    place.add_child(PlanToPose("PlanClear", BB.TARGET_POSE,
                              offset=(0, 0, RobotConfig.SAFE_HEIGHT)))
    place.add_child(ExecuteTrajectory())
    
    return place


def create_behavior_tree(pattern_type, pattern_value):
    """
    Create behavior tree for pick-and-place manipulation
    Compatible with py_trees 0.7.6
    """
    
    root = py_trees.composites.Sequence("Root")
    
    # Initialize
    root.add_child(InitializeTask(pattern_type, pattern_value))
    
    # Main loop
    main_loop = py_trees.composites.Sequence("MainLoop")
    
    # Setup phase
    setup_phase = py_trees.composites.Sequence("Setup")
    setup_phase.add_child(PlanToHome())
    setup_phase.add_child(ExecuteTrajectory())
    setup_phase.add_child(PerceiveCubes())
    setup_phase.add_child(SelectNextGoal())
    setup_phase.add_child(SelectBestCube())
    main_loop.add_child(setup_phase)
    
    retry = RetryDecorator()
    # Pick with retry
    pick_with_retry = RetryDecorator(
        name="RetryPick",
        child=create_pick_sequence(),
        num_failures=RobotConfig.MAX_GRASP_RETRIES
    )
    main_loop.add_child(pick_with_retry)
    
    # Place with retry
    place_with_retry = RetryDecorator(
        name="RetryPlace",
        child=create_place_sequence(),
        num_failures=RobotConfig.MAX_PLAN_RETRIES
    )
    main_loop.add_child(place_with_retry)
    
    root.add_child(main_loop)
    
    return root


###############################################################################
# MAIN
###############################################################################

def main():
    rospy.init_node('orchestrator_bt_advanced')
    
    # Parse parameters
    pattern_str = rospy.get_param('~pattern_type', 'STACK').upper()
    pattern_value = rospy.get_param('~pattern_value', '3')
    
    try:
        pattern_type = TaskPattern[pattern_str]
    except KeyError:
        rospy.logerr(f"Invalid pattern type: {pattern_str}")
        rospy.logerr(f"Valid options: {[p.value for p in TaskPattern]}")
        return
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("ADVANCED ORCHESTRATOR - Reactive Behavior Tree")
    rospy.loginfo(f"Task: {pattern_type.value} ({pattern_value})")
    rospy.loginfo(f"py_trees version: 0.7.6 compatible")
    rospy.loginfo("=" * 60)
    
    # Initialize blackboard
    init_blackboard()
    
    # Build tree
    root = create_behavior_tree(pattern_type, pattern_value)
    tree = py_trees_ros.trees.BehaviourTree(root)
    
    # Setup
    rospy.loginfo("Setting up behavior tree...")
    if not tree.setup(timeout=30):
        rospy.logerr("Tree setup failed!")
        return
    
    rospy.loginfo("Tree initialized. Starting execution...")
    
    # Run
    try:
        tree.tick_tock(sleep_ms=100)
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        tree.shutdown()


if __name__ == '__main__':
    main()

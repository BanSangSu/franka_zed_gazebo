#!/usr/bin/env python3
"""
Orchestrator Node - Behavior Tree with Custom Retry Decorator
Compatible with py_trees 0.7.6 (ROS Noetic)
Includes custom Retry decorator for robustness
"""


import rospy
import py_trees
import py_trees_ros
import actionlib
import numpy as np
import moveit_commander
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



# def switch_to_impedance():
#     rospy.wait_for_service('/controller_manager/switch_controller')
#     switcher = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
#     # stop position, start impedance
#     switcher(start_controllers=['joint_impedance_controller'], 
#              stop_controllers=['position_joint_trajectory_controller'],
#              strictness=2)
    
# class SwitchController(py_trees.behaviour.Behaviour):
#     """
#     Modes: 'IMPEDANCE' or 'EFFORT'
#     """
#     def __init__(self, mode):
#         super(SwitchController, self).__init__(f"SwitchTo{mode}")
#         self.mode = mode.upper()
#         # self.srv_name = '/group_rheinrobot/controller_manager/switch_controller' # for ROS_NAMESPACE
#         self.srv_name = '/controller_manager/switch_controller'
#         # We don't initialize the proxy here to avoid crashes if the service isn't up yet
#         self.srv = None

#     def setup(self, timeout):
#         """Standard py_trees setup to wait for ROS services"""
#         try:
#             rospy.wait_for_service(self.srv_name, timeout=timeout)
#             # CRITICAL: Use the Parent Service Class, NOT the Request class
#             self.srv = rospy.ServiceProxy(
#                 self.srv_name, 
#                 controller_manager_msgs.srv.SwitchController
#             )
#             return True
#         except rospy.ROSException as e:
#             rospy.logerr(f"[{self.name}] Service not found: {e}")
#             return False

#     def update(self):
#         # Create the request object using the Request class
#         req = controller_manager_msgs.srv.SwitchControllerRequest()
        
#         if self.mode == "IMPEDANCE":
#             req.start_controllers = ['cartesian_impedance_example_controller']
#             req.stop_controllers = ['position_joint_trajectory_controller']
#         else:
#             req.start_controllers = ['position_joint_trajectory_controller']
#             req.stop_controllers = ['cartesian_impedance_example_controller']
        
#         # Use the constant from the Request class
#         req.strictness = controller_manager_msgs.srv.SwitchControllerRequest.STRICT
        
#         try:
#             # The proxy (self.srv) was created using the Service Definition
#             res = self.srv(req)
#             return py_trees.common.Status.SUCCESS if res.ok else py_trees.common.Status.FAILURE
#         except Exception as e:
#             rospy.logerr(f"[{self.name}] Switch failed: {e}")
#             return py_trees.common.Status.FAILURE




###############################################################################
# CONFIGURATION
###############################################################################


class Config:
   """Robot configuration"""

   BASE_X = 0.5  ##what is its use?
   BASE_Y = 0.0
   BASE_Z = 0.0025 ##how accurate is this?
   CUBE_SIZE = 0.045
  
   SAFE_HEIGHT = 0.2
   GRASP_HEIGHT = -0.01
   PLACE_HEIGHT = 0.025
  
   GRIPPER_OPEN_WIDTH = 0.08
   GRIPPER_FORCE = 30.0
   GRIPPER_WIDTH_MARGIN = 0.005
   MIN_GRIPPER_WIDTH = 0.01 ##is it possible to tel, is it making a difference
  
   GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
  
   # Retry configuration
   MAX_PICK_RETRIES = 2
   MAX_PLACE_RETRIES = 3

   


class Pattern(Enum):
   """Task patterns"""
   STACK = "STACK"
   PYRAMID = "PYRAMID"
   LINE = "LINE"




###############################################################################
# CUSTOM RETRY DECORATOR
###############################################################################


class Retry(py_trees.decorators.Decorator):
   """
   Custom Retry Decorator for py_trees 0.7.6
  
   Retries the child behavior on FAILURE up to num_failures times.
   After max retries, returns FAILURE.
  
   Args:
       name: Decorator name
       child: Child behavior to retry
       num_failures: Maximum number of retry attempts (default: 3)
   """
  
   def __init__(self, name, child, num_failures=3):
       super(Retry, self).__init__(name=name, child=child)
       self.num_failures = num_failures
       self.count = 0
  
   def initialise(self):
       """Reset retry counter when decorator starts"""
       self.count = 0
  
   def update(self):
       """
       Retry logic:
       - If child succeeds: return SUCCESS
       - If child fails: retry if under limit, else return FAILURE
       - If child running: return RUNNING  ##How does it change to either success or faliure
       """
       child_status = self.decorated.status
      
       if child_status == py_trees.common.Status.SUCCESS:
           self.feedback_message = "Child succeeded"
           return py_trees.common.Status.SUCCESS
      
       elif child_status == py_trees.common.Status.FAILURE:
           self.count += 1
          
           if self.count < self.num_failures:
               # Reset child and retry
               rospy.logwarn(f"[{self.name}] Retry {self.count}/{self.num_failures}")
               self.decorated.stop(py_trees.common.Status.INVALID)
               self.decorated.setup(timeout=15)
               self.feedback_message = f"Retry {self.count}/{self.num_failures}"
               return py_trees.common.Status.RUNNING
           else:
               # Max retries exhausted
               rospy.logerr(f"[{self.name}] Failed after {self.num_failures} attempts")
               self.feedback_message = f"Failed after {self.num_failures} attempts"
               return py_trees.common.Status.FAILURE
      
       else:  # RUNNING
           return py_trees.common.Status.RUNNING




###############################################################################
# BLACKBOARD
###############################################################################


class BB:
   """Blackboard keys"""
   CUBES = "cubes"
   GOALS = "goals"
   GOAL_IDX = "goal_idx"
   COMPLETED = "completed"
   SELECTED_CUBE = "cube"
   TARGET_POSE = "target"
   TRAJECTORY = "traj"
   COLLISION = "collision"
   GOAL_TO_CUBE = "goal_to_cube"
   PLACED_IDS = "placed_cubes"

##maybe can add the gripper distance to check if it is holding an object


def init_blackboard():
   """Initialize blackboard"""
   bb = py_trees.blackboard.Blackboard()
   bb.set(BB.CUBES, [])
   bb.set(BB.GOALS, [])
   bb.set(BB.GOAL_IDX, 0)
   bb.set(BB.COMPLETED, set())
   bb.set(BB.TRAJECTORY, None)
   bb.set(BB.COLLISION, "")
   bb.set(BB.GOAL_TO_CUBE, {})
   bb.set(BB.PLACED_IDS, {})
   return bb




###############################################################################
# UTILITIES
###############################################################################


def quat_mult(q1, q2):
   """Multiply quaternions [x,y,z,w]"""
   x1, y1, z1, w1 = q1
   x2, y2, z2, w2 = q2
   return [
       w1*x2 + x1*w2 + y1*z2 - z1*y2,
       w1*y2 - x1*z2 + y1*w2 + z1*x2,
       w1*z2 + x1*y2 - y1*x2 + z1*w2,
       w1*w2 - x1*x2 - y1*y2 - z1*z2
   ]




def make_pose(x, y, z, quat=None):
   """Create Pose"""
   p = Pose()
   p.position.x, p.position.y, p.position.z = x, y, z
   if quat:
       p.orientation.x, p.orientation.y = quat[0], quat[1]
       p.orientation.z, p.orientation.w = quat[2], quat[3]
   else:
       p.orientation.w = 1.0
   return p




def offset_pose(pose, dx=0, dy=0, dz=0, quat=None):
   """Offset pose"""
   p = deepcopy(pose)
   p.position.x += dx
   p.position.y += dy
   p.position.z += dz
  
   if quat:
       orig = [pose.orientation.x, pose.orientation.y,
               pose.orientation.z, pose.orientation.w]
       new = quat_mult(orig, quat)
       p.orientation.x, p.orientation.y = new[0], new[1]
       p.orientation.z, p.orientation.w = new[2], new[3]
  
   return p




###############################################################################
# PATTERN GENERATORS
###############################################################################


def generate_stack(n):
   """Vertical stack"""
   goals = []
   for i in range(int(n)):
       goals.append(make_pose(
           Config.BASE_X,
           Config.BASE_Y,
           Config.BASE_Z + i * Config.CUBE_SIZE ##is this good enough check
       ))
   return goals




def generate_pyramid():
   """3-2-1 pyramid"""
   goals = []
   x, y, z = Config.BASE_X, Config.BASE_Y, Config.BASE_Z
   cs = Config.CUBE_SIZE ##should there be a slight offset
  
   for yo in [-cs, 0, cs]:
       goals.append(make_pose(x, y + yo, z))
   for yo in [-cs/2, cs/2]:
       goals.append(make_pose(x, y + yo, z + cs))
   goals.append(make_pose(x, y, z + 2*cs))
  
   return goals




def generate_line(n):
   """Horizontal line"""
   goals = []
   x, y, z = Config.BASE_X, Config.BASE_Y, Config.BASE_Z
   cs = Config.CUBE_SIZE
   start_y = y - (int(n) - 1) * cs / 2
  
   for i in range(int(n)):
       goals.append(make_pose(x, start_y + i * cs, z))
  
   return goals




###############################################################################
# BEHAVIORS
###############################################################################


class Perceive(py_trees.behaviour.Behaviour):
   """Scan workspace for cubes"""
  
   def __init__(self):
       super(Perceive, self).__init__("Perceive")
       self.bb = py_trees.blackboard.Blackboard()
       self.proxy = None
       self.scene = moveit_commander.PlanningSceneInterface()

  
   def setup(self, timeout):
       try:
           rospy.wait_for_service('/perception_service', timeout=timeout)
           self.proxy = rospy.ServiceProxy('/perception_service', PerceptionService)
           rospy.loginfo(f"[{self.name}] Ready")
           return True
       except:
           return False
  
   def update(self):
       try:
           resp = self.proxy(trigger=True)
           if not resp.success:
               return py_trees.common.Status.FAILURE
          
           placed_dict = self.bb.get(BB.PLACED_IDS) or {}
           placed_positions = list(placed_dict.values())
            
            # NOt sure how good this threshold is
           THRESHOLD = 0.05 

           cubes = []
           for i in range(resp.num_cubes):
            new_pos = resp.cube_poses.poses[i].position
            
            is_duplicate = False
            for p_pos in placed_positions:
                dist = np.sqrt((new_pos.x - p_pos.x)**2 + 
                                (new_pos.y - p_pos.y)**2 + 
                                (new_pos.z - p_pos.z)**2)
                if dist < THRESHOLD:
                    is_duplicate = True
                    self.scene.remove_world_object(f"cube_{i}")
                    break
            
            if not is_duplicate:
                cubes.append({
                    'id': i,
                    'pose': resp.cube_poses.poses[i],
                    'dimensions': resp.dimensions[i]
                })
            else:
                rospy.loginfo(f"Filtering out perceived cube at {new_pos.x, new_pos.y} - Space already occupied by a placed cube.")

           self.bb.set(BB.CUBES, cubes)
           rospy.loginfo(f"[{self.name}] Found {len(cubes)} cubes")
           return py_trees.common.Status.SUCCESS
       except Exception as e:
           rospy.logerr(f"[{self.name}] Error: {e}")
           return py_trees.common.Status.FAILURE




class InitGoals(py_trees.behaviour.Behaviour):
   """Initialize goal poses"""
  
   def __init__(self, pattern, value):
       super(InitGoals, self).__init__("InitGoals")
       self.pattern = pattern
       self.value = value
       self.bb = py_trees.blackboard.Blackboard()
  
   def update(self):
       if self.bb.get(BB.GOALS):
           return py_trees.common.Status.SUCCESS
      
       if self.pattern == Pattern.STACK:
           goals = generate_stack(self.value)
       elif self.pattern == Pattern.PYRAMID:
           goals = generate_pyramid()
       elif self.pattern == Pattern.LINE:
           goals = generate_line(self.value)
       else:
           goals = generate_stack(3)
      
       self.bb.set(BB.GOALS, goals)
       self.bb.set(BB.COMPLETED, set())
       self.bb.set(BB.GOAL_IDX, 0)
      
       rospy.loginfo(f"[{self.name}] Created {len(goals)} goals")
       return py_trees.common.Status.SUCCESS




class SelectGoal(py_trees.behaviour.Behaviour):
   """Select next incomplete goal"""
  
   def __init__(self):
       super(SelectGoal, self).__init__("SelectGoal")
       self.bb = py_trees.blackboard.Blackboard()
  
   def update(self):
       goals = self.bb.get(BB.GOALS)
       done = self.bb.get(BB.COMPLETED)
      
       for i, pose in enumerate(goals):
           if i not in done:
               self.bb.set(BB.GOAL_IDX, i)
               self.bb.set(BB.TARGET_POSE, pose)
               rospy.loginfo(f"[{self.name}] Goal {i+1}/{len(goals)}")
               return py_trees.common.Status.SUCCESS
      
       rospy.loginfo(f"[{self.name}] All goals complete!")
       return py_trees.common.Status.FAILURE




class SelectCube(py_trees.behaviour.Behaviour):
   """Select best available cube"""
  
   def __init__(self):
       super(SelectCube, self).__init__("SelectCube")
       self.bb = py_trees.blackboard.Blackboard()
  
   def update(self):
       cubes = self.bb.get(BB.CUBES)
       done = self.bb.get(BB.COMPLETED)
      
    #    available = [c for c in cubes if c['id'] not in done]
       available = cubes
      
       if not available:
           rospy.logwarn(f"[{self.name}] No cubes available")
           return py_trees.common.Status.FAILURE
      
       best = max(available, key=lambda c: c['pose'].position.x) ##unsure is this is a good heuristic, depending on where we thing x is zero it might also help in the solving the grabbing the aame cube twice
       self.bb.set(BB.SELECTED_CUBE, best)
      
       rospy.loginfo(f"[{self.name}] Selected cube {best['id']}")
       return py_trees.common.Status.SUCCESS




class SetCollision(py_trees.behaviour.Behaviour):
   """Set allowed collision for stacking"""
  
   def __init__(self):
       super(SetCollision, self).__init__("SetCollision")
       self.bb = py_trees.blackboard.Blackboard()
  
   def update(self):
       idx = self.bb.get(BB.GOAL_IDX)
       mapping = self.bb.get(BB.GOAL_TO_CUBE)
      
       collision = ""
       # Allow collision with the cube placed at the previous goal position
       if idx > 0 and (idx - 1) in mapping:
           prev_cube_id = mapping[idx - 1]
           collision = f"placed_cube_{prev_cube_id}"
           rospy.loginfo(f"[{self.name}] Allow interaction with: {collision}")
      
       self.bb.set(BB.COLLISION, collision)
       return py_trees.common.Status.SUCCESS




class MarkComplete(py_trees.behaviour.Behaviour):
   """Mark goal complete"""
  
   def __init__(self):
       super(MarkComplete, self).__init__("MarkComplete")
       self.bb = py_trees.blackboard.Blackboard()
  
   def update(self):
       idx = self.bb.get(BB.GOAL_IDX)
       cube = self.bb.get(BB.SELECTED_CUBE)
       target_pose = self.bb.get(BB.TARGET_POSE)
       placed_ids = self.bb.get(BB.PLACED_IDS) or {}
       # Update completion status
       done = self.bb.get(BB.COMPLETED)
       done.add(idx)
       self.bb.set(BB.COMPLETED, done)
      
       # Mapping for naming (placed_cube_ID)
       mapping = self.bb.get(BB.GOAL_TO_CUBE)
       mapping[idx] = cube['id']
       self.bb.set(BB.GOAL_TO_CUBE, mapping)

       ##added for placed logic

       placed_ids[idx] = target_pose.position 
       self.bb.set(BB.PLACED_IDS, placed_ids)
      
       return py_trees.common.Status.SUCCESS




       return py_trees.common.Status.SUCCESS




class CheckAttached(py_trees.behaviour.Behaviour):
   """
   Check if any objects are attached to the robot.
   """
   def __init__(self):
       super(CheckAttached, self).__init__("CheckAttached")
       self.scene = moveit_commander.PlanningSceneInterface()
  
   def update(self):
       # Returns SUCCESS if objects are attached, FAILURE otherwise
       attached = self.scene.get_attached_objects()
       if attached:
           ##rospy.loginfo(f"[{self.name}] Found attached objects: {list(attached.keys())}")
           return py_trees.common.Status.SUCCESS
       return py_trees.common.Status.FAILURE




class FindSafeDropPose(py_trees.behaviour.Behaviour):
   """
   Find an empty spot on the table for safe dropping.
   Iterates through candidate locations and checks for proximity to detected cubes
   AND already placed cubes.
   """
  
   def __init__(self):
       super(FindSafeDropPose, self).__init__("FindSafeDropPose")
       self.bb = py_trees.blackboard.Blackboard()
      
       # Grid of candidate spots on the table (x, y)
       self.candidates = [
           (0.3, 0.4), (0.3, -0.4), (0.5, 0.4), (0.5, -0.4),
           (0.4, 0.2), (0.4, -0.2), (0.6, 0.2), (0.6, -0.2),
           (0.3, 0.0), (0.7, 0.0)
       ]
  
   def update(self):
       detected_cubes = self.bb.get(BB.CUBES) or []
       mapping = self.bb.get(BB.GOAL_TO_CUBE) or {}
       goals = self.bb.get(BB.GOALS) or []
      
       # Collect positions of all cubes (detected and placed)
       occupied_positions = []
       for cube in detected_cubes:
           occupied_positions.append(cube['pose'].position)
      
       # Also consider goal locations that are marked complete
       done_indices = self.bb.get(BB.COMPLETED) or set()
       for idx in done_indices:
           if idx < len(goals):
               occupied_positions.append(goals[idx].position)
      
       # Search for a spot that is at least 15cm away from any cube
       safe_margin = 0.15
      
       found_spot = None
       for cx, cy in self.candidates:
           is_safe = True
           for pos in occupied_positions:
               dist = np.sqrt((pos.x - cx)**2 + (pos.y - cy)**2)
               if dist < safe_margin:
                   is_safe = False
                   break
          
           if is_safe:
               found_spot = (cx, cy)
               break
      
       if found_spot:
           cx, cy = found_spot
           rospy.loginfo(f"[{self.name}] Found safe spot at x={cx:.2f}, y={cy:.2f}")
       else:
           cx, cy = 0.3, 0.5 # Corner fallback
           rospy.logwarn(f"[{self.name}] No safe spot found! Using fallback x={cx:.2f}, y={cy:.2f}")
          
       # Use simple identity orientation for the target, PlanToPose will apply GRASP_QUAT
       drop_pose = make_pose(cx, cy, Config.BASE_Z + 0.05)
       self.bb.set(BB.TARGET_POSE, drop_pose)
      
       return py_trees.common.Status.SUCCESS




###############################################################################
# ACTION BEHAVIORS
###############################################################################


class ActionClient(py_trees.behaviour.Behaviour):
   """Base action client"""
  
   def __init__(self, name, action_name, action_type):
       super(ActionClient, self).__init__(name)
       self.action_name = action_name
       self.action_type = action_type
       self.client = None
       self.sent = False
       self.bb = py_trees.blackboard.Blackboard()
  
   def setup(self, timeout):
       self.client = actionlib.SimpleActionClient(self.action_name, self.action_type)
       if self.client.wait_for_server(rospy.Duration(timeout)):
           rospy.loginfo(f"[{self.name}] Ready")
           return True
       return False
  
   def initialise(self):
       self.sent = False
  
   def update(self):
       if not self.sent:
           goal = self.make_goal()
           if goal is None:
               return py_trees.common.Status.FAILURE
           self.client.send_goal(goal)
           self.sent = True
           return py_trees.common.Status.RUNNING
      
       state = self.client.get_state()
       if state == GoalStatus.SUCCEEDED:
           return self.on_success(self.client.get_result())
       elif state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
           return py_trees.common.Status.FAILURE
       return py_trees.common.Status.RUNNING
  
   def make_goal(self):
       raise NotImplementedError
  
   def on_success(self, result):
       return py_trees.common.Status.SUCCESS




class PlanHome(ActionClient):
   def __init__(self):
       super(PlanHome, self).__init__("PlanHome", "/planning_action", PlanningActionAction)
  
   def make_goal(self):
       g = PlanningActionGoal()
       g.action = "HOME"
       return g
  
   def on_success(self, result):
       if result and result.success:
           self.bb.set(BB.TRAJECTORY, result.trajectory)
           return py_trees.common.Status.SUCCESS
       return py_trees.common.Status.FAILURE



#added mode for normal or cartesian
class PlanToPose(ActionClient):
   def __init__(self, name, source, offset, collision=False, mode=""):
       super(PlanToPose, self).__init__(name, "/planning_action", PlanningActionAction)
       self.source = source
       self.offset = offset
       self.collision = collision
       self.mode = mode
  
   def make_goal(self):
       src = self.bb.get(self.source)
       placed_obj = self.bb.get(BB.PLACED_IDS)
       idx = self.bb.get(BB.GOAL_IDX)
       mapping = self.bb.get(BB.GOAL_TO_CUBE)
       g = PlanningActionGoal()
       g.action = self.mode
      
       if self.source == BB.SELECTED_CUBE:
           base = src['pose']
           g.allowed_collision_object = f"cube_{src['id']}"
       else:
           base = src
           if self.collision:
               g.allowed_collision_object = self.bb.get(BB.COLLISION)
               if idx > 0 and (idx - 1) in mapping:
                   rospy.loginfo(f"LIST OF COLLSIONS----------: [{self.bb.get(BB.COLLISION)}]")
                   rospy.loginfo(f"LIST OF COLLSIONS:ARE THESE THE SAME??----------: [{placed_obj}]")
                   rospy.loginfo(f"THE ALLOWED COLLISION IS---------:[{placed_obj[idx-1]}]") 
      
       target = offset_pose(base, *self.offset, quat=Config.GRASP_QUAT)
       g.target_pose = target
       rospy.loginfo(f"THE TARGET IS:(THE OFFSET SHOULD BE)---------:[{target}]")
       return g
  
   def on_success(self, result):
       if result and result.success:
           self.bb.set(BB.TRAJECTORY, result.trajectory)
           return py_trees.common.Status.SUCCESS
       return py_trees.common.Status.FAILURE




class Attach(ActionClient):
   def __init__(self):
       super(Attach, self).__init__("Attach", "/planning_action", PlanningActionAction)
  
   def make_goal(self):
       cube = self.bb.get(BB.SELECTED_CUBE)
       g = PlanningActionGoal()
       g.action = "ATTACH"
       g.object_name = f"cube_{cube['id']}"
       return g
  
   def on_success(self, result):
       return py_trees.common.Status.SUCCESS if result.success else py_trees.common.Status.FAILURE




class Detach(ActionClient):
   def __init__(self, target=None):
       super(Detach, self).__init__("Detach", "/planning_action", PlanningActionAction)
       self.target = target
  
   def make_goal(self):
       g = PlanningActionGoal()
       g.action = "DETACH"
      
       if self.target == "all":
           g.object_name = "all"
       else:
           cube = self.bb.get(BB.SELECTED_CUBE)
           if cube:
               g.object_name = f"cube_{cube['id']}"
           else:
               g.object_name = "all" # Fallback to all if no cube selected
       return g




class Execute(ActionClient):
   def __init__(self):
       super(Execute, self).__init__("Execute", "/control_action", ControlActionAction)
  
   def make_goal(self):
       traj = self.bb.get(BB.TRAJECTORY)
       if traj is None:
           rospy.logwarn("No trajectory sadly.......")
           return None
       g = ControlActionGoal()
       g.trajectory = traj
       return g




class GripperOpen(ActionClient):
   def __init__(self):
       super(GripperOpen, self).__init__("Open", "/gripper_action", GripperActionAction)
  
   def make_goal(self):
       g = GripperActionGoal()
       g.open = True
       g.width = Config.GRIPPER_OPEN_WIDTH
       return g
  
   def on_success(self, result):
       return py_trees.common.Status.SUCCESS




class GripperClose(ActionClient):
   def __init__(self):
       super(GripperClose, self).__init__("Close", "/gripper_action", GripperActionAction)
  
   def make_goal(self):
       cube = self.bb.get(BB.SELECTED_CUBE)
       dims = cube['dimensions']
       dx = dims.x if hasattr(dims, 'x') else dims[0]
       dy = dims.y if hasattr(dims, 'y') else dims[1]
      
       g = GripperActionGoal()
       g.open = False
       g.width = max(Config.MIN_GRIPPER_WIDTH, min(dx, dy) - Config.GRIPPER_WIDTH_MARGIN)
       g.force = Config.GRIPPER_FORCE
       return g
  
   def on_success(self, result):
       if result and result.success:
           return py_trees.common.Status.SUCCESS
       rospy.logwarn(f"[{self.name}] Grasp failed")
       return py_trees.common.Status.FAILURE




###############################################################################
# TREE CONSTRUCTION
###############################################################################


def build_pick_sequence():
   """Build pick subtree"""
   pick = py_trees.composites.Sequence("Pick")
   pick.add_child(GripperOpen())
   pick.add_child(PlanToPose("Approach", BB.SELECTED_CUBE, (0, 0, Config.SAFE_HEIGHT))) ##can also add the rotation of the cube that will be picked
   pick.add_child(Execute())
   pick.add_child(PlanToPose("Descend", BB.SELECTED_CUBE, (0, 0, Config.GRASP_HEIGHT), mode="BEAST"))
   pick.add_child(Execute())
   pick.add_child(GripperClose())
   pick.add_child(Attach())
   pick.add_child(PlanToPose("Lift", BB.SELECTED_CUBE, (0, 0, Config.SAFE_HEIGHT)))
   pick.add_child(Execute())
   return pick




def build_safe_drop_sequence():
   """Sequence to move to a safe spot and release everything if stuck"""
   drop = py_trees.composites.Sequence("SafeDrop")
  
   # Ensure robot is at Home for clear perception
   drop.add_child(PlanHome())
   drop.add_child(Execute())
  
   drop.add_child(Perceive())
   drop.add_child(FindSafeDropPose())
   drop.add_child(PlanToPose("MoveToSafeSpot", BB.TARGET_POSE, (0, 0, 0.1))) # Approach
   drop.add_child(Execute())
   drop.add_child(PlanToPose("DescendToSafeSpot", BB.TARGET_POSE, (0, 0, 0))) # Descend
   drop.add_child(Execute())
   drop.add_child(GripperOpen())
   drop.add_child(Detach(target="all"))
   drop.add_child(PlanToPose("LiftFromSafeSpot", BB.TARGET_POSE, (0, 0, 0.1))) # Lift
   drop.add_child(Execute())
   return drop



 ###when placed the cube, there seems to be a slight increase in the z-axis suhc that it seems to be lsight floating in the air.
def build_place_sequence():
   """Build place subtree"""
   place = py_trees.composites.Sequence("Place")
   place.add_child(SetCollision())
   place.add_child(PlanToPose("ApproachGoal", BB.TARGET_POSE, (0, 0, Config.SAFE_HEIGHT), True))
   place.add_child(Execute())
   place.add_child(PlanToPose("DescendGoal", BB.TARGET_POSE, (0, 0, Config.PLACE_HEIGHT), True, mode="BEAST"))
   place.add_child(Execute())
   place.add_child(GripperOpen())
   place.add_child(Detach())
   place.add_child(MarkComplete())
   place.add_child(PlanToPose("Clear", BB.TARGET_POSE, (0, 0, Config.SAFE_HEIGHT)))
   place.add_child(Execute())
   return place




def build_tree(pattern, value):
   """Build behavior tree with retry decorators"""
  
   root = py_trees.composites.Sequence("Root")
   root.add_child(InitGoals(pattern, value))
  
   loop = py_trees.composites.Sequence("Loop")
  
   # Setup & Recovery
   setup = py_trees.composites.Sequence("Setup")
  
   # Check for stuck cubes/objects at the start (Safe Drop Only if needed)
   recovery = py_trees.composites.Selector("Recovery")
   # 1. If nothing attached, finish recovery immediately (Success)
   recovery.add_child(py_trees.decorators.Inverter(CheckAttached()))
   # 2. Otherwise run the full safe drop sequence
   recovery.add_child(build_safe_drop_sequence())
   setup.add_child(recovery)
  
   setup.add_child(PlanHome())
   setup.add_child(Execute())
   setup.add_child(Perceive())
   setup.add_child(SelectGoal())
   setup.add_child(SelectCube())
   loop.add_child(setup)
  
   # Pick with retry
   pick_with_retry = Retry(
       name="RetryPick",
       child=build_pick_sequence(),
       num_failures=Config.MAX_PICK_RETRIES
   )
   loop.add_child(pick_with_retry)
  
   # Place with retry
   place_with_retry = Retry(
       name="RetryPlace",
       child=build_place_sequence(),
       num_failures=Config.MAX_PLACE_RETRIES
   )
   loop.add_child(place_with_retry)
  
   root.add_child(loop)
   return root




###############################################################################
# MAIN
###############################################################################


def main():
   rospy.init_node('orchestrator_bt')
  
   pattern_str = rospy.get_param('~pattern_type', 'STACK').upper()
   value = rospy.get_param('~pattern_value', '3')
  
   try:
       pattern = Pattern[pattern_str]
   except:
       rospy.logerr(f"Invalid pattern: {pattern_str}")
       return
  
   rospy.loginfo("=" * 60)
   rospy.loginfo("ORCHESTRATOR - BT with Retry Decorator")
   rospy.loginfo(f"Task: {pattern.value} ({value})")
   rospy.loginfo(f"Pick retries: {Config.MAX_PICK_RETRIES}")
   rospy.loginfo(f"Place retries: {Config.MAX_PLACE_RETRIES}")
   rospy.loginfo("=" * 60)
  
   init_blackboard()
  
   root = build_tree(pattern, value)
   tree = py_trees_ros.trees.BehaviourTree(root)
  
   if not tree.setup(timeout=30):
       rospy.logerr("Setup failed!")
       return
  
   rospy.loginfo("Starting...")
  
   try:
       tree.tick_tock(sleep_ms=100)
   except KeyboardInterrupt:
       pass
   finally:
       tree.shutdown()




if __name__ == '__main__':
   main()





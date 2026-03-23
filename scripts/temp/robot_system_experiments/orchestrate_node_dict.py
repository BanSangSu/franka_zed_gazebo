#!/usr/bin/env python3
"""
COMPLETE Fixed Orchestrator - Dynamic Stacking + Collision Handling
Compatible with py_trees 0.7.6 (ROS Noetic)
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

###############################################################################
# CONFIGURATION - FIXED HEIGHTS
###############################################################################

class Config:
   """Robot configuration - TUNED for stacking"""
   BASE_X = 0.5
   BASE_Y = 0.0
   BASE_Z = 0.0025  # Table height
   CUBE_SIZE = 0.045
   ID_MATCH_THRESHOLD = 0.03
   
   SAFE_HEIGHT = 0.2
   GRASP_HEIGHT = -0.0145  # Below cube center for grasp
   PLACE_HEIGHT = 0.005   # Slightly below target for firm placement
   
   GRIPPER_OPEN_WIDTH = 0.08
   GRIPPER_FORCE = 30.0
   GRIPPER_WIDTH_MARGIN = 0.005
   MIN_GRIPPER_WIDTH = 0.01
   
   GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]  # Vertical grasp
   
   MAX_PICK_RETRIES = 2
   MAX_PLACE_RETRIES = 3

class Pattern(Enum):
   STACK = "STACK"

###############################################################################
# CUSTOM RETRY DECORATOR
###############################################################################

class Retry(py_trees.decorators.Decorator):
   def __init__(self, name, child, num_failures=3):
       super(Retry, self).__init__(name=name, child=child)
       self.num_failures = num_failures
       self.count = 0
   
   def initialise(self):
       self.count = 0
   
   def update(self):
       child_status = self.decorated.status
       
       if child_status == py_trees.common.Status.SUCCESS:
           return py_trees.common.Status.SUCCESS
       elif child_status == py_trees.common.Status.FAILURE:
           self.count += 1
           if self.count < self.num_failures:
               rospy.logwarn(f"[{self.name}] Retry {self.count}/{self.num_failures}")
               self.decorated.stop(py_trees.common.Status.INVALID)
               self.decorated.setup(timeout=15)
               return py_trees.common.Status.RUNNING
           else:
               rospy.logerr(f"[{self.name}] Failed after {self.num_failures} attempts")
               return py_trees.common.Status.FAILURE
       return py_trees.common.Status.RUNNING

###############################################################################
# BLACKBOARD
###############################################################################

class BB:
   CUBES = "cubes"
   NEXT_ID = "next_id"
   GOALS = "goals"
   GOAL_IDX = "goal_idx"
   COMPLETED = "completed"
   SELECTED_CUBE = "cube"
   TARGET_POSE = "target"
   TRAJECTORY = "traj"
   COLLISION = "collision"
   GOAL_TO_CUBE = "goal_to_cube"

def init_blackboard():
   bb = py_trees.blackboard.Blackboard()
   bb.set(BB.CUBES, {})
   bb.set(BB.NEXT_ID, 0)
#    bb.set(BB.GOALS, [])
   bb.set(BB.GOAL_IDX, 0)
   bb.set(BB.COMPLETED, set())
   bb.set(BB.TRAJECTORY, None)
   bb.set(BB.COLLISION, "")
   bb.set(BB.GOAL_TO_CUBE, {})
   return bb

###############################################################################
# UTILITIES
###############################################################################

def quat_mult(q1, q2):
   x1, y1, z1, w1 = q1
   x2, y2, z2, w2 = q2
   return [
       w1*x2 + x1*w2 + y1*z2 - z1*y2,
       w1*y2 - x1*z2 + y1*w2 + z1*x2,
       w1*z2 + x1*y2 - y1*x2 + z1*w2,
       w1*w2 - x1*x2 - y1*y2 - z1*z2
   ]

def make_pose(x, y, z, quat=None):
   p = Pose()
   p.position.x, p.position.y, p.position.z = x, y, z
   if quat:
       p.orientation.x, p.orientation.y = quat[0], quat[1]
       p.orientation.z, p.orientation.w = quat[2], quat[3]
   else:
       p.orientation.w = 1.0
   return p

def offset_pose(pose, dx=0, dy=0, dz=0, quat=None):
   p = deepcopy(pose)
   p.position.x += dx
   p.position.y += dy
   p.position.z += dz
   if quat:
       orig = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
       new = quat_mult(orig, quat)
       p.orientation.x, p.orientation.y = new[0], new[1]
       p.orientation.z, p.orientation.w = new[2], new[3]
   return p

###############################################################################
# FIXED BEHAVIORS - DYNAMIC STACKING + COLLISION
###############################################################################

class Perceive(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(Perceive, self).__init__("Perceive")
       self.bb = py_trees.blackboard.Blackboard()
       self.proxy = None
   
   def setup(self, timeout):
       try:
           rospy.wait_for_service('/perception_service', timeout=timeout)
           self.proxy = rospy.ServiceProxy('/perception_service', PerceptionService)
           return True
       except:
           return False
   
   def get_dist(self, p1, p2):
       return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)
   
   def update(self):
       try:
           resp = self.proxy(trigger=True)
           if not resp.success:
               return py_trees.common.Status.FAILURE
           
           old_cubes = self.bb.get(BB.CUBES) or {}
           next_id = self.bb.get(BB.NEXT_ID) or 0
           
           # Parse detections
           current_detections = []
           for i in range(resp.num_cubes):
               current_detections.append({
                   'id': -1,
                   'pose': resp.cube_poses.poses[i],
                   'dimensions': resp.dimensions[i],
                   'label': resp.labels[i] if i < len(resp.labels) else "cube"
               })
           
           new_tracked_cubes = {}
           for det in current_detections:
               matched_id = None
               min_dist = Config.ID_MATCH_THRESHOLD
            
               for cube_id, old in old_cubes.items():
                   dist = self.get_dist(det['pose'].position, old['pose'].position)
                   if dist < min_dist:
                       min_dist = dist
                       matched_id = cube_id
               
               if matched_id:
                   det['id'] = matched_id
               else:
                   det['id'] = next_id
                   next_id += 1
               
               new_tracked_cubes[det['id']] = det
           
           self.bb.set(BB.CUBES, new_tracked_cubes)
           self.bb.set(BB.NEXT_ID, next_id)
           rospy.loginfo(f"[Perceive] Tracked {len(new_tracked_cubes)} cubes")
           return py_trees.common.Status.SUCCESS
       except Exception as e:
           rospy.logerr(f"[Perceive] Error: {e}")
           return py_trees.common.Status.FAILURE

class DynamicStackGoal(py_trees.behaviour.Behaviour):
   """DYNAMIC stack height from actual perception"""
   def __init__(self):
       super(DynamicStackGoal, self).__init__("DynamicStackGoal")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
        cubes_dict = self.bb.get(BB.CUBES) or {}
        mapping = self.bb.get(BB.GOAL_TO_CUBE) or {}
        
        # SINGLE LINE - Perfect for dictionary {id: cube_data}
        placed_cubes = [cubes_dict[cid] for cid in mapping.values() if cid in cubes_dict]
        
        if not placed_cubes:
            target_z = Config.BASE_Z + Config.CUBE_SIZE / 2
        else:
            highest = max(placed_cubes, key=lambda c: c['pose'].position.z)
            target_z = highest['pose'].position.z + Config.CUBE_SIZE
        
        target_pose = make_pose(Config.BASE_X, Config.BASE_Y, target_z)
        self.bb.set(BB.TARGET_POSE, target_pose)
        self.bb.set(BB.GOAL_IDX, len(mapping))
        
        rospy.loginfo(f"[DynamicStackGoal] Stack #{len(mapping)} at Z={target_z:.4f}")
        return py_trees.common.Status.SUCCESS

class RobustSetCollision(py_trees.behaviour.Behaviour):
   """FIXED collision - always allow collision with last placed cube"""
   def __init__(self):
       super(RobustSetCollision, self).__init__("SetCollision")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       mapping = self.bb.get(BB.GOAL_TO_CUBE) or {}
       if mapping:
           # Use LAST placed cube for collision
           last_cube_id = max(mapping.values())
           collision = f"placed_cube_{last_cube_id}"
           rospy.loginfo(f"[SetCollision] Allow collision with: {collision}")
       else:
           collision = ""
           rospy.loginfo("[SetCollision] No previous cubes, no collision allowance")
       
       self.bb.set(BB.COLLISION, collision)
       return py_trees.common.Status.SUCCESS

class SelectCube(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(SelectCube, self).__init__("SelectCube")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       cubes = self.bb.get(BB.CUBES) or {}
       mapping = self.bb.get(BB.GOAL_TO_CUBE) or {}
       
       # Filter out placed cubes       
       available_ids = [cid for cid in cubes if cid not in mapping.values()]
       if not available_ids:
           rospy.logwarn("[SelectCube] No available cubes!")
           return py_trees.common.Status.FAILURE
       
       best_id = max(available_ids, key=lambda cid: cubes[cid]['pose'].position.x)
       self.bb.set(BB.SELECTED_CUBE, cubes[best_id])
       rospy.loginfo(f"[SelectCube] Selected cube {cubes[best_id]['id']} at X={cubes[best_id]['pose'].position.x:.3f}")
       return py_trees.common.Status.SUCCESS

class MarkComplete(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(MarkComplete, self).__init__("MarkComplete")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       idx = self.bb.get(BB.GOAL_IDX)
       cube = self.bb.get(BB.SELECTED_CUBE)
       
       done = self.bb.get(BB.COMPLETED) or set()
       done.add(idx)
       self.bb.set(BB.COMPLETED, done)
       
       mapping = self.bb.get(BB.GOAL_TO_CUBE) or {}
       mapping[idx] = cube['id']
       self.bb.set(BB.GOAL_TO_CUBE, mapping)
       
       rospy.loginfo(f"[MarkComplete] Goal {idx} → Cube {cube['id']}")
       return py_trees.common.Status.SUCCESS

###############################################################################
# ACTION CLIENTS (UNCHANGED)
###############################################################################

class ActionClient(py_trees.behaviour.Behaviour):
   def __init__(self, name, action_name, action_type):
       super(ActionClient, self).__init__(name)
       self.action_name = action_name
       self.action_type = action_type
       self.client = None
       self.sent = False
       self.bb = py_trees.blackboard.Blackboard()
   
   def setup(self, timeout):
       self.client = actionlib.SimpleActionClient(self.action_name, self.action_type)
       return self.client.wait_for_server(rospy.Duration(timeout))
   
   def initialise(self):
       self.sent = False
   
   def update(self):
       if not self.sent:
           goal = self.make_goal()
           if goal is None: return py_trees.common.Status.FAILURE
           self.client.send_goal(goal)
           self.sent = True
           return py_trees.common.Status.RUNNING
       
       state = self.client.get_state()
       if state == GoalStatus.SUCCEEDED:
           return self.on_success(self.client.get_result())
       elif state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
           return py_trees.common.Status.FAILURE
       return py_trees.common.Status.RUNNING

class PlanHome(ActionClient):
   def __init__(self): super(PlanHome, self).__init__("PlanHome", "/planning_action", PlanningActionAction)
   def make_goal(self):
       g = PlanningActionGoal()
       g.action = "HOME"
       return g
   def on_success(self, result): 
       if result and result.success: self.bb.set(BB.TRAJECTORY, result.trajectory)
       return py_trees.common.Status.SUCCESS if result and result.success else py_trees.common.Status.FAILURE

class PlanToPose(ActionClient):
   def __init__(self, name, source, offset, collision=False):
       super(PlanToPose, self).__init__(name, "/planning_action", PlanningActionAction)
       self.source = source
       self.offset = offset
       self.collision = collision
   
   def make_goal(self):
       src = self.bb.get(self.source)
       g = PlanningActionGoal()
       g.action = ""
       
       if self.source == BB.SELECTED_CUBE:
           base = src['pose']
           g.allowed_collision_object = f"cube_{src['id']}"
       else:
           base = src
           if self.collision:
               collision_obj = self.bb.get(BB.COLLISION)
               g.allowed_collision_object = collision_obj if collision_obj else ""
               rospy.loginfo(f"[PlanToPose] Collision object: '{g.allowed_collision_object}'")
       
       target = offset_pose(base, *self.offset, quat=Config.GRASP_QUAT)
       g.target_pose = target
       rospy.loginfo(f"[PlanToPose] Target Z: {target.position.z:.4f}")
       return g
   
   def on_success(self, result):
       if result and result.success:
           self.bb.set(BB.TRAJECTORY, result.trajectory)
           return py_trees.common.Status.SUCCESS
       return py_trees.common.Status.FAILURE

class Attach(ActionClient):
   def __init__(self): super(Attach, self).__init__("Attach", "/planning_action", PlanningActionAction)
   def make_goal(self):
       cube = self.bb.get(BB.SELECTED_CUBE)
       g = PlanningActionGoal()
       g.action = "ATTACH"
       g.object_name = f"cube_{cube['id']}"
       return g
   def on_success(self, result): return py_trees.common.Status.SUCCESS if result and result.success else py_trees.common.Status.FAILURE

class Detach(ActionClient):
    def __init__(self): 
        super(Detach, self).__init__("Detach", "/planning_action", PlanningActionAction)
    
    def make_goal(self):
        g = PlanningActionGoal()
        g.action = "DETACH"
        cube = self.bb.get(BB.SELECTED_CUBE)
        g.object_name = f"cube_{cube['id']}" if cube else "all"
        return g
    
    def on_success(self, result):
        rospy.loginfo("[Detach] ✓ Detached object")
        
        # UPDATE POSITION with TARGET_POSE (final placed location)
        cube = self.bb.get(BB.SELECTED_CUBE)
        target_pose = self.bb.get(BB.TARGET_POSE)
        
        if cube and target_pose:
            cube_id = cube['id']
            cubes_dict = self.bb.get(BB.CUBES) or {}
            
            # Update position in dictionary {id: cube_data}
            cubes_dict[cube_id]['pose'] = target_pose
            self.bb.set(BB.CUBES, cubes_dict)
            
            rospy.loginfo(f"[Detach] Updated cube {cube_id} position to target Z={target_pose.position.z:.4f}")
        
        return py_trees.common.Status.SUCCESS

class Execute(ActionClient):
    def __init__(self): 
        super(Execute, self).__init__("Execute", "/control_action", ControlActionAction)
    
    def make_goal(self):
        traj = self.bb.get(BB.TRAJECTORY)
        if traj is None: return None
        g = ControlActionGoal()
        g.trajectory = traj
        return g
    
    def on_success(self, result): 
        rospy.loginfo("[Execute] Trajectory executed successfully")
        return py_trees.common.Status.SUCCESS


class GripperOpen(ActionClient):
   def __init__(self): super(GripperOpen, self).__init__("GripperOpen", "/gripper_action", GripperActionAction)
   def make_goal(self):
       g = GripperActionGoal()
       g.open = True
       g.width = Config.GRIPPER_OPEN_WIDTH
       return g
   def on_success(self, result):
        # FIXED: Don't check result.success - just log
        rospy.loginfo("[GripperOpen] ✓ Opened successfully")
        return py_trees.common.Status.SUCCESS

class GripperClose(ActionClient):
   def __init__(self): 
       super(GripperClose, self).__init__("GripperClose", "/gripper_action", GripperActionAction)
       self.initial_width = None
   
   def initialise(self):
       super(GripperClose, self).initialise()
       self.initial_width = None
   
   def make_goal(self):
       cube = self.bb.get(BB.SELECTED_CUBE)
       dims = cube['dimensions']
       dx = dims.x if hasattr(dims, 'x') else dims[0]
       dy = dims.y if hasattr(dims, 'y') else dims[1]
       
       g = GripperActionGoal()
       g.open = False
       g.width = max(Config.MIN_GRIPPER_WIDTH, min(dx, dy) - Config.GRIPPER_WIDTH_MARGIN)
       g.force = Config.GRIPPER_FORCE
       
       # Store expected grasp width BEFORE closing
       self.expected_width = g.width
       rospy.loginfo(f"[GripperClose] Target: {g.width:.4f}m (cube size)")
       return g
   
   def on_success(self, result):
       if not result or not result.success:
           rospy.logwarn(f"[GripperClose] Action FAILED: {result.message}")
           return py_trees.common.Status.FAILURE
       
       final_width = result.final_width
       rospy.loginfo(f"[GripperClose] final_width={final_width:.4f}m (expected={self.expected_width:.4f}m)")
       
       # ✅ GRASP SUCCESS CONDITIONS:
       # 1. Stopped near expected cube width (±5mm tolerance)
       width_match = abs(final_width - self.expected_width) < 0.005
       # 2. OR significantly closed from open position
       closed_enough = final_width < 0.030  
       
       if width_match or closed_enough:
           rospy.loginfo(f"✅ GRASP SUCCESS (match={width_match}, closed={closed_enough})")
           return py_trees.common.Status.SUCCESS
       else:
           rospy.logwarn(f"❌ EMPTY CLOSE (final={final_width:.4f}m vs expected={self.expected_width:.4f}m)")
           return py_trees.common.Status.FAILURE

###############################################################################
# BEHAVIOR TREE - FIXED STRUCTURE
###############################################################################

def build_pick_sequence():
   pick = py_trees.composites.Sequence("Pick")
   pick.add_child(GripperOpen())
   pick.add_child(PlanToPose("Approach", BB.SELECTED_CUBE, (0, 0, Config.SAFE_HEIGHT)))
   pick.add_child(Execute())
   pick.add_child(PlanToPose("Descend", BB.SELECTED_CUBE, (0, 0, Config.GRASP_HEIGHT)))
   pick.add_child(Execute())
   pick.add_child(GripperClose())
   pick.add_child(Attach())
   pick.add_child(PlanToPose("Lift", BB.SELECTED_CUBE, (0, 0, Config.SAFE_HEIGHT)))
   pick.add_child(Execute())
   return pick

def build_place_sequence():
   place = py_trees.composites.Sequence("Place")
   place.add_child(RobustSetCollision())  # ← FIXED collision
   place.add_child(PlanToPose("Approach", BB.TARGET_POSE, (0, 0, Config.SAFE_HEIGHT), collision=True))
   place.add_child(Execute())
   place.add_child(PlanToPose("Descend", BB.TARGET_POSE, (0, 0, Config.PLACE_HEIGHT), collision=True))
   place.add_child(Execute())
   place.add_child(Detach())  # Detach BEFORE gripper open
   place.add_child(GripperOpen())
   place.add_child(MarkComplete())
   place.add_child(PlanToPose("Clear", BB.TARGET_POSE, (0, 0, Config.SAFE_HEIGHT), collision=True))
   place.add_child(Execute())
   return place

def build_tree():
   root = py_trees.composites.Sequence("Root")
   
   # Main loop
   loop = py_trees.composites.Sequence("MainLoop")
   
   # Recovery + Setup
   setup = py_trees.composites.Sequence("Setup")
   setup.add_child(PlanHome())
   setup.add_child(Execute())
   setup.add_child(Perceive())
   setup.add_child(DynamicStackGoal())  # ← DYNAMIC stacking
   setup.add_child(SelectCube())
   loop.add_child(setup)
   
   # Pick + Place with retries
   loop.add_child(Retry("RetryPick", build_pick_sequence(), Config.MAX_PICK_RETRIES))
   loop.add_child(Retry("RetryPlace", build_place_sequence(), Config.MAX_PLACE_RETRIES))
   
   root.add_child(loop)
   return root

###############################################################################
# MAIN
###############################################################################

def main():
   rospy.init_node('orchestrator_bt')
   
   rospy.loginfo("=" * 60)
   rospy.loginfo("COMPLETE FIXED ORCHESTRATOR")
   rospy.loginfo("Dynamic stacking + Robust collision handling")
   rospy.loginfo("=" * 60)
   
   init_blackboard()
   root = build_tree()
   tree = py_trees_ros.trees.BehaviourTree(root)
   
   if not tree.setup(timeout=120):
       rospy.logerr("Tree setup failed!")
       return
   
   rospy.loginfo("Starting dynamic stacking...")
   try:
       tree.tick_tock(sleep_ms=100)
   except KeyboardInterrupt:
       pass
   finally:
       pass

if __name__ == '__main__':
   main()

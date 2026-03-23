#!/usr/bin/env python3
"""
FIXED Orchestrator - Argument Order Corrected for py_trees 0.7.6
Robust Spatial Filtering + ID Exclusion
"""

import rospy
import py_trees
import py_trees_ros
import actionlib
import numpy as np
from copy import deepcopy
from geometry_msgs.msg import Pose
from franka_zed_gazebo.srv import PerceptionService
from franka_zed_gazebo.msg import (
   ControlActionAction, ControlActionGoal,
   GripperActionAction, GripperActionGoal,
   PlanningActionAction, PlanningActionGoal
)

class Config:
   BASE_X = 0.5
   BASE_Y = 0.0
   BASE_Z = 0.0025
   CUBE_SIZE = 0.045
   
   # Radius to ignore cubes already on the stack
   STACK_XY_TOLERANCE = 0.10 
   
   SAFE_HEIGHT = 0.2
   GRASP_HEIGHT = -0.0145
   PLACE_HEIGHT = 0.005
   
   GRIPPER_OPEN_WIDTH = 0.08
   GRIPPER_FORCE = 30.0
   GRIPPER_WIDTH_MARGIN = 0.005
   MIN_GRIPPER_WIDTH = 0.01
   GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
   
   MAX_PICK_RETRIES = 2
   MAX_PLACE_RETRIES = 3

class BB:
   CUBES = "cubes"
   GOAL_IDX = "goal_idx"
   SELECTED_CUBE = "cube"
   TARGET_POSE = "target"
   TRAJECTORY = "traj"
   USED_IDS = "used_ids"

def init_blackboard():
   bb = py_trees.blackboard.Blackboard()
   bb.set(BB.CUBES, {})
   bb.set(BB.GOAL_IDX, 0)
   bb.set(BB.USED_IDS, set())
   bb.set(BB.TRAJECTORY, None)
   return bb

###############################################################################
# CUSTOM RETRY DECORATOR (FIXED FOR 0.7.6)
###############################################################################

class Retry(py_trees.decorators.Decorator):
   def __init__(self, name, child, num):
       # CRITICAL: In py_trees 0.7.6, Decorator init is (child, name)
       super(Retry, self).__init__(child=child, name=name)
       self.num = num
       self.count = 0
   
   def initialise(self):
       self.count = 0
   
   def update(self):
       # Access the decorated child behavior
       child_status = self.decorated.status
       
       if child_status == py_trees.common.Status.SUCCESS:
           return py_trees.common.Status.SUCCESS
       
       if child_status == py_trees.common.Status.FAILURE:
           self.count += 1
           if self.count < self.num:
               rospy.logwarn(f"[{self.name}] Retry {self.count}/{self.num}")
               # Reset the child so it can run again
               self.decorated.stop(py_trees.common.Status.INVALID)
               return py_trees.common.Status.RUNNING
           else:
               rospy.logerr(f"[{self.name}] Max retries reached.")
               return py_trees.common.Status.FAILURE
               
       return py_trees.common.Status.RUNNING

###############################################################################
# BEHAVIORS
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
       except: return False
   
   def update(self):
       try:
           resp = self.proxy(trigger=True)
           if not resp.success: return py_trees.common.Status.FAILURE
           detected_cubes = {}
           for i in range(resp.num_cubes):
               detected_cubes[i] = {'id': i, 'pose': resp.cube_poses.poses[i], 'dimensions': resp.dimensions[i]}
           self.bb.set(BB.CUBES, detected_cubes)
           return py_trees.common.Status.SUCCESS
       except: return py_trees.common.Status.FAILURE

class SelectCube(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(SelectCube, self).__init__("SelectCube")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       cubes = self.bb.get(BB.CUBES) or {}
       used_ids = self.bb.get(BB.USED_IDS) or set()
       candidates = []
       
       for cid, data in cubes.items():
           pos = data['pose'].position
           dist = np.sqrt((pos.x - Config.BASE_X)**2 + (pos.y - Config.BASE_Y)**2)
           
           if dist > Config.STACK_XY_TOLERANCE and cid not in used_ids:
               candidates.append(data)
           else:
               rospy.loginfo(f"[SelectCube] Skipping cube_{cid} (dist={dist:.3f}m, used={cid in used_ids})")

       if not candidates:
           rospy.logwarn("[SelectCube] No valid cubes found!")
           return py_trees.common.Status.FAILURE
       
       best_cube = max(candidates, key=lambda c: c['pose'].position.x)
       self.bb.set(BB.SELECTED_CUBE, best_cube)
       rospy.loginfo(f"[SelectCube] ✅ Selected Cube_{best_cube['id']}")
       return py_trees.common.Status.SUCCESS

class MarkComplete(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(MarkComplete, self).__init__("MarkComplete")
       self.bb = py_trees.blackboard.Blackboard()
   def update(self):
       idx = self.bb.get(BB.GOAL_IDX) or 0
       self.bb.set(BB.GOAL_IDX, idx + 1)
       sel = self.bb.get(BB.SELECTED_CUBE)
       if sel:
           used = self.bb.get(BB.USED_IDS) or set()
           used.add(sel['id'])
           self.bb.set(BB.USED_IDS, used)
       return py_trees.common.Status.SUCCESS

# --- Standard Action Classes (Helper Functions Included) ---

def make_pose(x, y, z, quat=None):
   p = Pose(); p.position.x, p.position.y, p.position.z = x, y, z
   if quat: p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
   else: p.orientation.w = 1.0
   return p

def offset_pose(pose, dx, dy, dz, quat=None):
   p = deepcopy(pose); p.position.x += dx; p.position.y += dy; p.position.z += dz
   if quat: p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
   return p

class ActionClient(py_trees.behaviour.Behaviour):
   def __init__(self, name, action_name, action_type):
       super(ActionClient, self).__init__(name)
       self.action_name = action_name
       self.action_type = action_type
       self.client = None
       self.sent = False
       self.bb = py_trees.blackboard.Blackboard()
   def setup(self, t):
       self.client = actionlib.SimpleActionClient(self.action_name, self.action_type)
       return self.client.wait_for_server(rospy.Duration(t))
   def initialise(self): self.sent = False
   def update(self):
       if not self.sent:
           goal = self.make_goal()
           if not goal: return py_trees.common.Status.FAILURE
           self.client.send_goal(goal); self.sent = True; return py_trees.common.Status.RUNNING
       if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED: return self.on_success(self.client.get_result())
       if self.client.get_state() in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]: return py_trees.common.Status.FAILURE
       return py_trees.common.Status.RUNNING

class PlanHome(ActionClient):
   def __init__(self): super(PlanHome, self).__init__("PlanHome", "/planning_action", PlanningActionAction)
   def make_goal(self): g = PlanningActionGoal(); g.action = "HOME"; return g
   def on_success(self, res): self.bb.set(BB.TRAJECTORY, res.trajectory); return py_trees.common.Status.SUCCESS

class PlanToPose(ActionClient):
   def __init__(self, name, source, offset, col=False):
       super(PlanToPose, self).__init__(name, "/planning_action", PlanningActionAction)
       self.source = source; self.offset = offset; self.col = col
   def make_goal(self):
       src = self.bb.get(self.source); g = PlanningActionGoal(); g.action = ""
       if self.source == BB.SELECTED_CUBE: base = src['pose']; g.allowed_collision_object = f"cube_{src['id']}"
       else: base = src; g.allowed_collision_object = "all" if self.col else ""
       g.target_pose = offset_pose(base, *self.offset, quat=Config.GRASP_QUAT)
       return g
   def on_success(self, res): self.bb.set(BB.TRAJECTORY, res.trajectory); return py_trees.common.Status.SUCCESS

class Execute(ActionClient):
    def __init__(self): super(Execute, self).__init__("Execute", "/control_action", ControlActionAction)
    def make_goal(self): traj = self.bb.get(BB.TRAJECTORY); g = ControlActionGoal(); g.trajectory = traj; return g if traj else None
    def on_success(self, res): return py_trees.common.Status.SUCCESS

class GripperClose(ActionClient):
   def __init__(self): super(GripperClose, self).__init__("GripperClose", "/gripper_action", GripperActionAction)
   def make_goal(self):
       c = self.bb.get(BB.SELECTED_CUBE); d = c['dimensions']; width = min(d.x, d.y) if hasattr(d, 'x') else min(d[0], d[1])
       g = GripperActionGoal(); g.open = False; g.width = max(Config.MIN_GRIPPER_WIDTH, width - Config.GRIPPER_WIDTH_MARGIN); g.force = Config.GRIPPER_FORCE; return g
   def on_success(self, res): return py_trees.common.Status.SUCCESS if res and res.final_width > 0.005 else py_trees.common.Status.FAILURE

class GripperOpen(ActionClient):
   def __init__(self): super(GripperOpen, self).__init__("GripperOpen", "/gripper_action", GripperActionAction)
   def make_goal(self): g = GripperActionGoal(); g.open = True; g.width = Config.GRIPPER_OPEN_WIDTH; return g
   def on_success(self, res): return py_trees.common.Status.SUCCESS

class Attach(ActionClient):
   def __init__(self): super(Attach, self).__init__("Attach", "/planning_action", PlanningActionAction)
   def make_goal(self): c = self.bb.get(BB.SELECTED_CUBE); g = PlanningActionGoal(); g.action = "ATTACH"; g.object_name = f"cube_{c['id']}"; return g
   def on_success(self, res): return py_trees.common.Status.SUCCESS

class Detach(ActionClient):
    def __init__(self): super(Detach, self).__init__("Detach", "/planning_action", PlanningActionAction)
    def make_goal(self): c = self.bb.get(BB.SELECTED_CUBE); g = PlanningActionGoal(); g.action = "DETACH"; g.object_name = f"cube_{c['id']}" if c else "all"; return g
    def on_success(self, res): return py_trees.common.Status.SUCCESS

class DynamicStackGoal(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(DynamicStackGoal, self).__init__("DynamicStackGoal")
       self.bb = py_trees.blackboard.Blackboard()
   def update(self):
        count = self.bb.get(BB.GOAL_IDX) or 0
        tz = Config.BASE_Z + (count * Config.CUBE_SIZE) + (Config.CUBE_SIZE / 2)
        self.bb.set(BB.TARGET_POSE, make_pose(Config.BASE_X, Config.BASE_Y, tz))
        return py_trees.common.Status.SUCCESS

###############################################################################
# TREE BUILDING
###############################################################################

def build_tree():
   root = py_trees.composites.Sequence("Root")
   loop = py_trees.composites.Sequence("MainLoop")
   
   setup = py_trees.composites.Sequence("Setup")
   setup.add_child(PlanHome())
   setup.add_child(Execute())
   setup.add_child(Perceive())
   setup.add_child(DynamicStackGoal())
   setup.add_child(SelectCube())
   loop.add_child(setup)
   
   pick = py_trees.composites.Sequence("PickSequence")
   pick.add_child(GripperOpen())
   pick.add_child(PlanToPose("ApproachPick", BB.SELECTED_CUBE, (0,0,Config.SAFE_HEIGHT)))
   pick.add_child(Execute())
   pick.add_child(PlanToPose("DescendPick", BB.SELECTED_CUBE, (0,0,Config.GRASP_HEIGHT)))
   pick.add_child(Execute())
   pick.add_child(GripperClose())
   pick.add_child(Attach())
   pick.add_child(PlanToPose("Lift", BB.SELECTED_CUBE, (0,0,Config.SAFE_HEIGHT)))
   pick.add_child(Execute())
   
   loop.add_child(Retry("RetryPick", pick, Config.MAX_PICK_RETRIES))
   
   place = py_trees.composites.Sequence("PlaceSequence")
   place.add_child(PlanToPose("ApproachPlace", BB.TARGET_POSE, (0,0,Config.SAFE_HEIGHT), True))
   place.add_child(Execute())
   place.add_child(PlanToPose("DescendPlace", BB.TARGET_POSE, (0,0,Config.PLACE_HEIGHT), True))
   place.add_child(Execute())
   place.add_child(Detach())
   place.add_child(GripperOpen())
   place.add_child(MarkComplete())
   place.add_child(PlanToPose("Clear", BB.TARGET_POSE, (0,0,Config.SAFE_HEIGHT), True))
   place.add_child(Execute())
   
   loop.add_child(Retry("RetryPlace", place, Config.MAX_PLACE_RETRIES))
   
   root.add_child(loop)
   return root

def main():
   rospy.init_node('orchestrator_bt')
   init_blackboard()
   tree = py_trees_ros.trees.BehaviourTree(build_tree())
   if tree.setup(timeout=30):
       rospy.loginfo("Tree setup successful. Starting...")
       tree.tick_tock(100)
   else:
       rospy.logerr("Tree setup failed.")

if __name__ == '__main__':
   main()
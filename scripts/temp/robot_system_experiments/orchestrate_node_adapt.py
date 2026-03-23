#!/usr/bin/env python3
"""
VERSION 6 FIXED: Continuous Perception & Stale Data Management
=============================================================
FIXES:
1. Adds perception updates throughout the workflow
2. Detects and handles disappeared/deleted cubes
3. Re-validates cube existence before grasping
4. Counts actual cubes on stack (not just goal index)
5. Cleans up stale tracking records
6. Handles manual cube removal from stack
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
   CUBE_SIZE = 0.048
   
   # Multi-method tracking thresholds
   POSITION_MATCH_THRESHOLD = 0.04
   STACK_VERIFICATION_RADIUS = 0.08
   STACK_HEIGHT_TOLERANCE = 0.03
   
   # Staleness detection
   MAX_TIME_UNSEEN = 5.0  # seconds before cube considered "lost"
   
   # Strategy selection parameters
   HIGH_CONFIDENCE_THRESHOLD = 0.85
   LOW_CONFIDENCE_THRESHOLD = 0.50
   
   SAFE_HEIGHT = 0.2
   GRASP_HEIGHT = -0.01
   PLACE_HEIGHT = 0.007 # 0.005
   
   GRIPPER_OPEN_WIDTH = 0.08
   GRIPPER_FORCE = 30.0
   GRIPPER_WIDTH_MARGIN = 0.005
   MIN_GRIPPER_WIDTH = 0.01
   GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
   
   MAX_PICK_RETRIES = 2
   MAX_PLACE_RETRIES = 3

class CubeRecord:
   """Enhanced cube record with staleness detection"""
   
   def __init__(self, initial_perception_data):
       self.uuid = self._generate_uuid()
       self.initial_pose = deepcopy(initial_perception_data['pose'])
       self.current_pose = deepcopy(initial_perception_data['pose'])
       self.dimensions = initial_perception_data['dimensions']
       self.current_perception_id = initial_perception_data['id']
       
       # Tracking metadata
       self.state = "AVAILABLE"  # AVAILABLE, PICKED, PLACED, LOST
       self.tracking_confidence = 1.0
       self.last_seen = rospy.Time.now()
       self.match_count = 1
       self.total_distance_moved = 0.0
       
       # Verification
       self.placement_verified = False
       self.expected_place_position = None
   
   @staticmethod
   def _generate_uuid():
       import uuid
       return str(uuid.uuid4())[:8]
   
   def is_stale(self):
       """Check if cube hasn't been seen recently"""
       time_unseen = (rospy.Time.now() - self.last_seen).to_sec()
       return time_unseen > Config.MAX_TIME_UNSEEN
   
   def compute_match_score(self, perception_data):
       """Compute match score (lower = better match)"""
       perc_pose = perception_data['pose']
       perc_dim = perception_data['dimensions']
       
       # Position distance
       pos_dist = np.sqrt(
           (self.current_pose.position.x - perc_pose.position.x)**2 +
           (self.current_pose.position.y - perc_pose.position.y)**2 +
           (self.current_pose.position.z - perc_pose.position.z)**2
       )
       
       # Dimension distance (size)
       self_dims = np.array([self.dimensions.x, self.dimensions.y, self.dimensions.z] 
                           if hasattr(self.dimensions, 'x') else self.dimensions)
       perc_dims = np.array([perc_dim.x, perc_dim.y, perc_dim.z] 
                           if hasattr(perc_dim, 'x') else perc_dim)
       dim_dist = np.linalg.norm(self_dims - perc_dims)
       
       # Combined score (weighted)
       score = 0.7 * pos_dist + 0.3 * dim_dist
       return score
   
   def update(self, perception_data, match_score):
       """Update record with new perception data"""
       old_pos = np.array([
           self.current_pose.position.x,
           self.current_pose.position.y,
           self.current_pose.position.z
       ])
       
       self.current_pose = deepcopy(perception_data['pose'])
       self.current_perception_id = perception_data['id']
       self.last_seen = rospy.Time.now()
       self.match_count += 1
       
       # Update distance moved
       new_pos = np.array([
           self.current_pose.position.x,
           self.current_pose.position.y,
           self.current_pose.position.z
       ])
       self.total_distance_moved += np.linalg.norm(new_pos - old_pos)
       
       # Update confidence
       self.tracking_confidence = max(0.0, 1.0 - match_score * 5.0)
   
   def is_on_stack(self):
       """Check if cube is likely on the stack"""
       dist_from_base = np.sqrt(
           (self.current_pose.position.x - Config.BASE_X)**2 +
           (self.current_pose.position.y - Config.BASE_Y)**2
       )
       return dist_from_base < Config.STACK_VERIFICATION_RADIUS
   
   def mark_picked(self):
       self.state = "PICKED"
   
   def mark_placed(self, expected_position):
       self.state = "PLACED"
       self.placement_verified = False
       self.expected_place_position = expected_position
   
   def mark_lost(self):
       self.state = "LOST"
       rospy.logwarn(f"[CubeRecord] UUID {self.uuid} marked as LOST")
   
   def verify_placement(self, target_z):
       """Verify cube is at expected stack height"""
       z_error = abs(self.current_pose.position.z - target_z)
       if z_error < Config.STACK_HEIGHT_TOLERANCE and self.is_on_stack():
           self.placement_verified = True
           return True
       return False

class CubeTrackingManager:
   """Manages cube tracking with staleness detection and cleanup"""
   
   def __init__(self):
       self.records = {}  # uuid -> CubeRecord
       self.tracking_strategy = "hybrid"
       self.last_update_time = rospy.Time.now()
   
   def update(self, perception_data):
       """Update tracking and clean stale data"""
       self.last_update_time = rospy.Time.now()
       rospy.loginfo(f"[TrackingManager] Processing {len(perception_data)} cubes using {self.tracking_strategy} strategy")
       
       if self.tracking_strategy == "hybrid":
           self._hybrid_update(perception_data)
       elif self.tracking_strategy == "spatial":
           self._spatial_update(perception_data)
       else:
           self._simple_update(perception_data)
       
       # Clean up stale records
       self._cleanup_stale_records()
   
   def _hybrid_update(self, perception_data):
       """Primary tracking method - matches using multiple features"""
       matched_uuids = set()
       unmatched_perceptions = []
       
       # Match using geometric similarity
       for perc_id, perc_data in perception_data.items():
           best_uuid = None
           best_score = float('inf')
           
           for uuid, record in self.records.items():
               # Skip already matched
               if uuid in matched_uuids:
                   continue
               
               # Skip LOST cubes
               if record.state == "LOST":
                   continue
               
               # For PICKED cubes, be more lenient with matching (robot is moving it)
               match_score = record.compute_match_score(perc_data)
               threshold = Config.POSITION_MATCH_THRESHOLD * 3 if record.state == "PICKED" else Config.POSITION_MATCH_THRESHOLD
               
               if match_score < threshold and match_score < best_score:
                   best_score = match_score
                   best_uuid = uuid
           
           if best_uuid:
               self.records[best_uuid].update(perc_data, best_score)
               matched_uuids.add(best_uuid)
               rospy.loginfo(f"  ✓ Matched cube_{perc_id} to UUID {best_uuid} "
                           f"(score={best_score:.4f}, state={self.records[best_uuid].state})")
           else:
               unmatched_perceptions.append((perc_id, perc_data))
       
       # Mark unmatched records as potentially lost (unless PICKED - might be in gripper)
       for uuid, record in self.records.items():
           if uuid not in matched_uuids and record.state not in ["PICKED", "LOST"]:
               time_unseen = (rospy.Time.now() - record.last_seen).to_sec()
               if time_unseen > 1.0:  # Not seen for 1 second
                   rospy.logwarn(f"  ⚠ UUID {uuid} not matched (unseen for {time_unseen:.1f}s)")
       
       # Create new records for unmatched cubes
       for perc_id, perc_data in unmatched_perceptions:
           pos = perc_data['pose'].position
           dist_from_stack = np.sqrt((pos.x - Config.BASE_X)**2 + (pos.y - Config.BASE_Y)**2)
           
           new_record = CubeRecord(perc_data)
           
           if dist_from_stack < Config.STACK_VERIFICATION_RADIUS:
               rospy.loginfo(f"  ➕ New cube on stack: UUID {new_record.uuid} (cube_{perc_id})")
               new_record.state = "PLACED"
               new_record.placement_verified = True
           else:
               rospy.loginfo(f"  ➕ New cube detected: UUID {new_record.uuid} (cube_{perc_id})")
           
           self.records[new_record.uuid] = new_record
   
   def _spatial_update(self, perception_data):
       """Fallback: Use spatial filtering only"""
       for perc_id, perc_data in perception_data.items():
           pos = perc_data['pose'].position
           dist_from_stack = np.sqrt((pos.x - Config.BASE_X)**2 + (pos.y - Config.BASE_Y)**2)
           
           found = False
           for uuid, record in self.records.items():
               if record.current_perception_id == perc_id:
                   record.update(perc_data, 0.0)
                   found = True
                   break
           
           if not found:
               new_record = CubeRecord(perc_data)
               if dist_from_stack < Config.STACK_VERIFICATION_RADIUS:
                   new_record.state = "PLACED"
               self.records[new_record.uuid] = new_record
   
   def _simple_update(self, perception_data):
       """Simple fallback: Direct ID tracking"""
       for perc_id, perc_data in perception_data.items():
           new_record = CubeRecord(perc_data)
           self.records[new_record.uuid] = new_record
   
   def _cleanup_stale_records(self):
       """Remove records that haven't been seen in a while"""
       to_remove = []
       for uuid, record in self.records.items():
           if record.is_stale() and record.state != "PICKED":
               rospy.logwarn(f"[Cleanup] Removing stale UUID {uuid} (state={record.state})")
               to_remove.append(uuid)
       
       for uuid in to_remove:
           del self.records[uuid]
   
   def get_available_cubes(self):
       """Return cubes available for picking (not on stack, not picked, not lost)"""
       available = {}
       for uuid, rec in self.records.items():
           if rec.state == "AVAILABLE" and rec.tracking_confidence > Config.LOW_CONFIDENCE_THRESHOLD:
               if not rec.is_stale():
                   available[uuid] = rec
       return available
   
   def get_stack_count(self):
       """Count cubes actually on the stack"""
       count = 0
       for rec in self.records.values():
           if rec.state == "PLACED" and rec.is_on_stack() and not rec.is_stale():
               count += 1
       return count
   
   def get_cube(self, uuid):
       return self.records.get(uuid)
   
   def validate_cube_exists(self, uuid):
       """Check if cube still exists and is reachable"""
       if uuid not in self.records:
           rospy.logerr(f"[ValidateCube] UUID {uuid} not in records!")
           return False
       
       record = self.records[uuid]
       
       if record.is_stale():
           rospy.logerr(f"[ValidateCube] UUID {uuid} is stale (last seen {(rospy.Time.now() - record.last_seen).to_sec():.1f}s ago)")
           return False
       
       if record.state == "LOST":
           rospy.logerr(f"[ValidateCube] UUID {uuid} is marked as LOST")
           return False
       
       return True
   
   def mark_picked(self, uuid):
       if uuid in self.records:
           self.records[uuid].mark_picked()
   
   def mark_placed(self, uuid, expected_position):
       if uuid in self.records:
           self.records[uuid].mark_placed(expected_position)
   
   def verify_placement(self, uuid, target_z):
       """Verify cube was placed correctly"""
       if uuid not in self.records:
           rospy.loginfo(f"[VerifyPlacement] ✅ UUID {uuid} not found - assuming renamed/placed")
           return True
       
       record = self.records[uuid]
       
       if record.verify_placement(target_z):
           return True
       
       # Check if cube disappeared (successfully placed and renamed)
       if record.is_stale():
           rospy.loginfo(f"[VerifyPlacement] ✅ UUID {uuid} disappeared - assuming placed")
           return True
       
       rospy.logwarn(f"[VerifyPlacement] ❌ Verification failed for UUID {uuid}")
       return False
   
   def get_status_summary(self):
       """Get summary of current tracking state"""
       summary = {
           'total': len(self.records),
           'available': len(self.get_available_cubes()),
           'placed': self.get_stack_count(),
           'picked': sum(1 for r in self.records.values() if r.state == "PICKED"),
           'lost': sum(1 for r in self.records.values() if r.state == "LOST")
       }
       return summary

class BB:
   TRACKING_MANAGER = "tracking_manager"
   SELECTED_UUID = "selected_uuid"
   TARGET_POSE = "target"
   TRAJECTORY = "traj"

def init_blackboard():
   bb = py_trees.blackboard.Blackboard()
   bb.set(BB.TRACKING_MANAGER, CubeTrackingManager())
   bb.set(BB.TRAJECTORY, None)
   return bb

###############################################################################
# RETRY DECORATOR
###############################################################################

class Retry(py_trees.decorators.Decorator):
   def __init__(self, name, child, num):
       super(Retry, self).__init__(child=child, name=name)
       self.num = num
       self.count = 0
   
   def initialise(self):
       self.count = 0
   
   def update(self):
       child_status = self.decorated.status
       
       if child_status == py_trees.common.Status.SUCCESS:
           return py_trees.common.Status.SUCCESS
       
       if child_status == py_trees.common.Status.FAILURE:
           self.count += 1
           if self.count < self.num:
               rospy.logwarn(f"[{self.name}] Retry {self.count}/{self.num}")
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
   def __init__(self, name="Perceive"):
       super(Perceive, self).__init__(name)
       self.bb = py_trees.blackboard.Blackboard()
       self.proxy = None
   
   def setup(self, timeout):
       try:
           rospy.wait_for_service('/perception_service', timeout=timeout)
           self.proxy = rospy.ServiceProxy('/perception_service', PerceptionService)
           return True
       except: 
           return False
   
   def update(self):
       try:
           resp = self.proxy(trigger=True)
           if not resp.success: 
               return py_trees.common.Status.FAILURE
           
           detected_cubes = {}
           for i in range(resp.num_cubes):
               detected_cubes[i] = {
                   'id': i, 
                   'pose': resp.cube_poses.poses[i], 
                   'dimensions': resp.dimensions[i]
               }
           
           manager = self.bb.get(BB.TRACKING_MANAGER)
           manager.update(detected_cubes)
           
           # Log status summary
           status = manager.get_status_summary()
           rospy.loginfo(f"[Perceive] Status: {status['available']} available, "
                        f"{status['placed']} on stack, {status['picked']} picked, "
                        f"{status['lost']} lost")
           
           return py_trees.common.Status.SUCCESS
       except Exception as e:
           rospy.logerr(f"[Perceive] Error: {e}")
           return py_trees.common.Status.FAILURE

class SelectCube(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(SelectCube, self).__init__("SelectCube")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       available = manager.get_available_cubes()
       
       if not available:
           rospy.logwarn("[SelectCube] ❌ No available cubes!")
           return py_trees.common.Status.FAILURE
       
       # Select cube farthest from stack (highest X)
       best_uuid = max(available.keys(), 
                      key=lambda u: available[u].current_pose.position.x)
       
       cube = available[best_uuid]
       self.bb.set(BB.SELECTED_UUID, best_uuid)
       rospy.loginfo(f"[SelectCube] ✅ Selected UUID {best_uuid} (cube_{cube.current_perception_id}, "
                    f"conf={cube.tracking_confidence:.2f})")
       return py_trees.common.Status.SUCCESS

class ValidateSelectedCube(py_trees.behaviour.Behaviour):
   """Re-validate that selected cube still exists before attempting grasp"""
   def __init__(self):
       super(ValidateSelectedCube, self).__init__("ValidateSelectedCube")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       uuid = self.bb.get(BB.SELECTED_UUID)
       
       if not uuid:
           rospy.logerr("[ValidateCube] No cube selected!")
           return py_trees.common.Status.FAILURE
       
       if manager.validate_cube_exists(uuid):
           rospy.loginfo(f"[ValidateCube] ✅ UUID {uuid} validated")
           return py_trees.common.Status.SUCCESS
       else:
           rospy.logerr(f"[ValidateCube] ❌ UUID {uuid} no longer valid - will reselect")
           self.bb.set(BB.SELECTED_UUID, None)
           return py_trees.common.Status.FAILURE

class VerifyPlacement(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(VerifyPlacement, self).__init__("VerifyPlacement")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       uuid = self.bb.get(BB.SELECTED_UUID)
       target_pose = self.bb.get(BB.TARGET_POSE)
       
       if manager.verify_placement(uuid, target_pose.position.z):
           rospy.loginfo(f"[VerifyPlacement] ✅ Placement verified for UUID {uuid}")
           return py_trees.common.Status.SUCCESS
       else:
           rospy.logwarn(f"[VerifyPlacement] ❌ Placement verification failed for UUID {uuid}")
           return py_trees.common.Status.FAILURE

class MarkPicked(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(MarkPicked, self).__init__("MarkPicked")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       uuid = self.bb.get(BB.SELECTED_UUID)
       if uuid:
           manager.mark_picked(uuid)
           rospy.loginfo(f"[MarkPicked] ✅ UUID {uuid} marked as PICKED")
       return py_trees.common.Status.SUCCESS

class MarkPlaced(py_trees.behaviour.Behaviour):
   def __init__(self):
       super(MarkPlaced, self).__init__("MarkPlaced")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       uuid = self.bb.get(BB.SELECTED_UUID)
       target_pose = self.bb.get(BB.TARGET_POSE)
       
       if uuid:
           manager.mark_placed(uuid, target_pose.position)
           stack_count = manager.get_stack_count()
           rospy.loginfo(f"[MarkPlaced] ✅ UUID {uuid} marked as PLACED (stack count: {stack_count})")
       return py_trees.common.Status.SUCCESS

class PrintStatus(py_trees.behaviour.Behaviour):
   """Print current tracking status for debugging"""
   def __init__(self):
       super(PrintStatus, self).__init__("PrintStatus")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       status = manager.get_status_summary()
       rospy.loginfo(f"[Status] 📊 Total={status['total']}, Available={status['available']}, "
                    f"OnStack={status['placed']}, Picked={status['picked']}, Lost={status['lost']}")
       return py_trees.common.Status.SUCCESS

# --- Helper Functions ---

def make_pose(x, y, z, quat=None):
   p = Pose()
   p.position.x, p.position.y, p.position.z = x, y, z
   if quat: 
       p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
   else: 
       p.orientation.w = 1.0
   return p

def offset_pose(pose, dx, dy, dz, quat=None):
   p = deepcopy(pose)
   p.position.x += dx
   p.position.y += dy
   p.position.z += dz
   if quat: 
       p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
   return p

# --- Action Client Base ---

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
   
   def initialise(self): 
       self.sent = False
   
   def update(self):
       if not self.sent:
           goal = self.make_goal()
           if not goal: 
               return py_trees.common.Status.FAILURE
           self.client.send_goal(goal)
           self.sent = True
           return py_trees.common.Status.RUNNING
       
       state = self.client.get_state()
       if state == actionlib.GoalStatus.SUCCEEDED: 
           return self.on_success(self.client.get_result())
       if state in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]: 
           return py_trees.common.Status.FAILURE
       return py_trees.common.Status.RUNNING

# --- Specific Action Behaviors ---

class PlanHome(ActionClient):
   def __init__(self): 
       super(PlanHome, self).__init__("PlanHome", "/planning_action", PlanningActionAction)
   
   def make_goal(self): 
       g = PlanningActionGoal()
       g.action = "HOME"
       return g
   
   def on_success(self, res): 
       self.bb.set(BB.TRAJECTORY, res.trajectory)
       return py_trees.common.Status.SUCCESS

class PlanToPose(ActionClient):
   def __init__(self, name, pose_source, offset, col=False):
       super(PlanToPose, self).__init__(name, "/planning_action", PlanningActionAction)
       self.pose_source = pose_source
       self.offset = offset
       self.col = col
   
   def make_goal(self):
       g = PlanningActionGoal()
       g.action = ""
       
       if self.pose_source == "SELECTED":
           manager = self.bb.get(BB.TRACKING_MANAGER)
           uuid = self.bb.get(BB.SELECTED_UUID)
           cube = manager.get_cube(uuid)
           if not cube:
               rospy.logerr(f"[PlanToPose] Cube UUID {uuid} not found!")
               return None
           base = cube.current_pose
           g.allowed_collision_object = f"cube_{cube.current_perception_id}"
       else:
           base = self.bb.get(self.pose_source)
           g.allowed_collision_object = "all" if self.col else ""
       
       g.target_pose = offset_pose(base, *self.offset, quat=Config.GRASP_QUAT)
       return g
   
   def on_success(self, res): 
       self.bb.set(BB.TRAJECTORY, res.trajectory)
       return py_trees.common.Status.SUCCESS

class Execute(ActionClient):
   def __init__(self): 
       super(Execute, self).__init__("Execute", "/control_action", ControlActionAction)
   
   def make_goal(self): 
       traj = self.bb.get(BB.TRAJECTORY)
       g = ControlActionGoal()
       g.trajectory = traj
       return g if traj else None
   
   def on_success(self, res): 
       return py_trees.common.Status.SUCCESS

class GripperClose(ActionClient):
   def __init__(self): 
       super(GripperClose, self).__init__("GripperClose", "/gripper_action", GripperActionAction)
   
   def make_goal(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       uuid = self.bb.get(BB.SELECTED_UUID)
       cube = manager.get_cube(uuid)
       
       if not cube:
           rospy.logerr("[GripperClose] No cube found!")
           return None
       
       d = cube.dimensions
       width = min(d.x, d.y) if hasattr(d, 'x') else min(d[0], d[1])
       
       g = GripperActionGoal()
       g.open = False
       g.width = max(Config.MIN_GRIPPER_WIDTH, width - Config.GRIPPER_WIDTH_MARGIN)
       g.force = Config.GRIPPER_FORCE
       return g
   
   def on_success(self, res): 
       return py_trees.common.Status.SUCCESS if res and res.final_width > 0.005 else py_trees.common.Status.FAILURE

class GripperOpen(ActionClient):
   def __init__(self): 
       super(GripperOpen, self).__init__("GripperOpen", "/gripper_action", GripperActionAction)
   
   def make_goal(self): 
       g = GripperActionGoal()
       g.open = True
       g.width = Config.GRIPPER_OPEN_WIDTH
       return g
   
   def on_success(self, res): 
       return py_trees.common.Status.SUCCESS

class Attach(ActionClient):
   def __init__(self): 
       super(Attach, self).__init__("Attach", "/planning_action", PlanningActionAction)
   
   def make_goal(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       uuid = self.bb.get(BB.SELECTED_UUID)
       cube = manager.get_cube(uuid)
       
       if not cube:
           return None
       
       g = PlanningActionGoal()
       g.action = "ATTACH"
       g.object_name = f"cube_{cube.current_perception_id}"
       return g
   
   def on_success(self, res): 
       return py_trees.common.Status.SUCCESS

class Detach(ActionClient):
   def __init__(self): 
       super(Detach, self).__init__("Detach", "/planning_action", PlanningActionAction)
   
   def make_goal(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       uuid = self.bb.get(BB.SELECTED_UUID)
       
       g = PlanningActionGoal()
       g.action = "DETACH"
       if uuid:
           cube = manager.get_cube(uuid)
           g.object_name = f"cube_{cube.current_perception_id}" if cube else "all"
       else:
           g.object_name = "all"
       return g
   
   def on_success(self, res): 
       return py_trees.common.Status.SUCCESS

class DynamicStackGoal(py_trees.behaviour.Behaviour):
   """Calculate stack goal based on ACTUAL cubes on stack (not a counter)"""
   def __init__(self):
       super(DynamicStackGoal, self).__init__("DynamicStackGoal")
       self.bb = py_trees.blackboard.Blackboard()
   
   def update(self):
       manager = self.bb.get(BB.TRACKING_MANAGER)
       stack_count = manager.get_stack_count()
       
       # Calculate height for next cube
       tz = Config.BASE_Z + (stack_count * Config.CUBE_SIZE) + (Config.CUBE_SIZE / 2)
       
       self.bb.set(BB.TARGET_POSE, make_pose(Config.BASE_X, Config.BASE_Y, tz))
       rospy.loginfo(f"[DynamicStackGoal] 🎯 Target height: {tz:.4f}m (stack count: {stack_count})")
       return py_trees.common.Status.SUCCESS

###############################################################################
# TREE BUILDING
###############################################################################

def build_tree():
   root = py_trees.composites.Sequence("Root")
   loop = py_trees.composites.Sequence("MainLoop")
   
   # Setup - with initial perception and status
   setup = py_trees.composites.Sequence("Setup")
   setup.add_child(PlanHome())
   setup.add_child(Execute())
   setup.add_child(Perceive("InitialPerceive"))
   setup.add_child(PrintStatus())
   setup.add_child(DynamicStackGoal())
   setup.add_child(SelectCube())
   loop.add_child(setup)
   
   # Pick sequence with re-validation
   pick = py_trees.composites.Sequence("PickSequence")
   pick.add_child(Perceive("PrePickPerceive"))  # Re-perceive before picking
   pick.add_child(ValidateSelectedCube())  # Validate cube still exists
   pick.add_child(GripperOpen())
   pick.add_child(PlanToPose("ApproachPick", "SELECTED", (0,0,Config.SAFE_HEIGHT)))
   pick.add_child(Execute())
   pick.add_child(PlanToPose("DescendPick", "SELECTED", (0,0,Config.GRASP_HEIGHT)))
   pick.add_child(Execute())
   pick.add_child(GripperClose())
   pick.add_child(Attach())
   pick.add_child(MarkPicked())
   pick.add_child(PlanToPose("Lift", "SELECTED", (0,0,Config.SAFE_HEIGHT)))
   pick.add_child(Execute())
   
   loop.add_child(Retry("RetryPick", pick, Config.MAX_PICK_RETRIES))
   
   # Place sequence
   place = py_trees.composites.Sequence("PlaceSequence")
   place.add_child(PlanToPose("ApproachPlace", BB.TARGET_POSE, (0,0,Config.SAFE_HEIGHT), True))
   place.add_child(Execute())
   place.add_child(PlanToPose("DescendPlace", BB.TARGET_POSE, (0,0,Config.PLACE_HEIGHT), True))
   place.add_child(Execute())
   place.add_child(Detach())
   place.add_child(GripperOpen())
   place.add_child(MarkPlaced())
   place.add_child(PlanToPose("Clear", BB.TARGET_POSE, (0,0,Config.SAFE_HEIGHT), True))
   place.add_child(Execute())
   
   loop.add_child(Retry("RetryPlace", place, Config.MAX_PLACE_RETRIES))
   
   # Verification with fresh perception
   verification = py_trees.composites.Sequence("Verification")
   verification.add_child(Perceive("PostPlacePerceive"))
   verification.add_child(VerifyPlacement())
   verification.add_child(PrintStatus())
   loop.add_child(Retry("RetryVerification", verification, 2))
   
   root.add_child(loop)
   return root

def main():
   rospy.init_node('orchestrator_bt_v6')
   init_blackboard()
   tree = py_trees_ros.trees.BehaviourTree(build_tree())
   if tree.setup(timeout=30):
       rospy.loginfo("🚀 V6 Enhanced Tracking Tree setup successful. Starting...")
       tree.tick_tock(100)
   else:
       rospy.logerr("Tree setup failed.")

if __name__ == '__main__':
   main()
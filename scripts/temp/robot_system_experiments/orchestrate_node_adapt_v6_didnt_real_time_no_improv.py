#!/usr/bin/env python3
"""
VERSION 6 IMPROVED: Robust Cube Removal Detection and Status Updates
====================================================================
IMPROVEMENTS:
1. Detects when cubes are removed from scene (both placed and unpicked)
2. Updates cube status to REMOVED when no longer detected
3. Tracks consecutive detection failures
4. Handles partial stack removal
5. Maintains accurate available cube count
6. Better logging and state transitions
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
   
   # Multi-method tracking thresholds
   POSITION_MATCH_THRESHOLD = 0.04
   STACK_VERIFICATION_RADIUS = 0.08
   STACK_HEIGHT_TOLERANCE = 0.03
   
   # Removal detection
   CONSECUTIVE_MISSES_FOR_REMOVAL = 2  # How many perception cycles before marking REMOVED
   MAX_TIME_UNSEEN = 5.0  # seconds
   
   # Strategy selection parameters
   HIGH_CONFIDENCE_THRESHOLD = 0.85
   LOW_CONFIDENCE_THRESHOLD = 0.50
   
   SAFE_HEIGHT = 0.2
   GRASP_HEIGHT = -0.0145
   PLACE_HEIGHT = 0.0
   
   GRIPPER_OPEN_WIDTH = 0.08
   GRIPPER_FORCE = 30.0
   GRIPPER_WIDTH_MARGIN = 0.005
   MIN_GRIPPER_WIDTH = 0.01
   GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
   
   MAX_PICK_RETRIES = 2
   MAX_PLACE_RETRIES = 3

class CubeRecord:
   """Enhanced cube record with removal detection"""
   
   def __init__(self, initial_perception_data):
       self.uuid = self._generate_uuid()
       self.initial_pose = deepcopy(initial_perception_data['pose'])
       self.current_pose = deepcopy(initial_perception_data['pose'])
       self.dimensions = initial_perception_data['dimensions']
       self.current_perception_id = initial_perception_data['id']
       
       # Tracking metadata
       self.state = "AVAILABLE"  # AVAILABLE, PICKED, PLACED, REMOVED
       self.tracking_confidence = 1.0
       self.last_seen = rospy.Time.now()
       self.match_count = 1
       self.total_distance_moved = 0.0
       
       # Removal detection
       self.consecutive_misses = 0
       self.was_ever_seen_after_placement = False
       
       # Verification
       self.placement_verified = False
       self.expected_place_position = None
   
   @staticmethod
   def _generate_uuid():
       import uuid
       return str(uuid.uuid4())[:8]
   
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
       
       # Dimension distance
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
       self.consecutive_misses = 0  # Reset miss counter
       
       # Track if we've seen this cube after placement
       if self.state == "PLACED":
           self.was_ever_seen_after_placement = True
       
       # Update distance moved
       new_pos = np.array([
           self.current_pose.position.x,
           self.current_pose.position.y,
           self.current_pose.position.z
       ])
       self.total_distance_moved += np.linalg.norm(new_pos - old_pos)
       
       # Update confidence
       self.tracking_confidence = max(0.0, 1.0 - match_score * 5.0)
   
   def mark_miss(self):
       """Mark that this cube was not detected in current perception cycle"""
       self.consecutive_misses += 1
       time_unseen = (rospy.Time.now() - self.last_seen).to_sec()
       
       # Check if we should mark as removed
       if (self.consecutive_misses >= Config.CONSECUTIVE_MISSES_FOR_REMOVAL or 
           time_unseen > Config.MAX_TIME_UNSEEN):
           if self.state != "REMOVED":
               old_state = self.state
               self.state = "REMOVED"
               rospy.logwarn(f"[CubeRecord] UUID {self.uuid} marked REMOVED "
                           f"(was {old_state}, misses={self.consecutive_misses}, "
                           f"unseen={time_unseen:.1f}s)")
               return True
       return False
   
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
       self.consecutive_misses = 0  # Reset for placement verification
   
   def verify_placement(self, target_z):
       """Verify cube is at expected stack height"""
       z_error = abs(self.current_pose.position.z - target_z)
       if z_error < Config.STACK_HEIGHT_TOLERANCE and self.is_on_stack():
           self.placement_verified = True
           return True
       return False

class CubeTrackingManager:
   """Manages cube tracking with removal detection"""
   
   def __init__(self):
       self.records = {}  # uuid -> CubeRecord
       self.perception_history = []
       self.tracking_strategy = "hybrid"
       self.initial_cube_count = 0
       self.placed_count = 0
       self.removed_count = 0
       self.last_perception_time = rospy.Time.now()
   
   def update(self, perception_data):
       """Update tracking and detect removals"""
       current_time = rospy.Time.now()
       time_since_last = (current_time - self.last_perception_time).to_sec()
       self.last_perception_time = current_time
       
       rospy.loginfo(f"[TrackingManager] Processing {len(perception_data)} cubes "
                    f"(Δt={time_since_last:.2f}s, strategy={self.tracking_strategy})")
       
       # Track initial cube count
       if self.initial_cube_count == 0:
           self.initial_cube_count = len(perception_data)
       
       # Update based on strategy
       if self.tracking_strategy == "hybrid":
           self._hybrid_update(perception_data)
       elif self.tracking_strategy == "spatial":
           self._spatial_update(perception_data)
       else:
           self._simple_update(perception_data)
       
       # Check for removed cubes
       self._detect_removals()
       
       # Log status
       self._log_status()
   
   def _hybrid_update(self, perception_data):
       """Primary tracking method with removal detection"""
       matched_uuids = set()
       unmatched_perceptions = []
       
       # Match using geometric similarity
       for perc_id, perc_data in perception_data.items():
           best_uuid = None
           best_score = float('inf')
           
           for uuid, record in self.records.items():
               # Skip removed cubes
               if record.state == "REMOVED":
                   continue
               
               # Skip cubes being manipulated (but allow matching PLACED cubes)
               if record.state == "PICKED":
                   continue
               
               match_score = record.compute_match_score(perc_data)
               
               if match_score < Config.POSITION_MATCH_THRESHOLD and match_score < best_score:
                   best_score = match_score
                   best_uuid = uuid
           
           if best_uuid:
               self.records[best_uuid].update(perc_data, best_score)
               matched_uuids.add(best_uuid)
               rospy.logdebug(f"  Matched cube_{perc_id} to UUID {best_uuid} "
                            f"(score={best_score:.4f}, state={self.records[best_uuid].state})")
           else:
               unmatched_perceptions.append((perc_id, perc_data))
       
       # Mark unmatched cubes as missed
       for uuid, record in self.records.items():
           if uuid not in matched_uuids and record.state != "REMOVED":
               record.mark_miss()
       
       # Create new records for unmatched cubes
       for perc_id, perc_data in unmatched_perceptions:
           pos = perc_data['pose'].position
           dist_from_stack = np.sqrt((pos.x - Config.BASE_X)**2 + (pos.y - Config.BASE_Y)**2)
           
           new_record = CubeRecord(perc_data)
           
           if dist_from_stack < Config.STACK_VERIFICATION_RADIUS:
               rospy.loginfo(f"  New cube on stack: UUID {new_record.uuid} (cube_{perc_id})")
               new_record.state = "PLACED"
               new_record.placement_verified = True
           else:
               rospy.loginfo(f"  New cube detected: UUID {new_record.uuid} (cube_{perc_id})")
           
           self.records[new_record.uuid] = new_record
       
       # Check if we should degrade strategy
       active_records = [r for r in self.records.values() if r.state != "REMOVED"]
       if active_records:
           avg_confidence = np.mean([r.tracking_confidence for r in active_records])
           if avg_confidence < Config.LOW_CONFIDENCE_THRESHOLD:
               rospy.logwarn(f"[TrackingManager] Low confidence ({avg_confidence:.2f}), "
                           f"degrading to spatial strategy")
               self.tracking_strategy = "spatial"
   
   def _spatial_update(self, perception_data):
       """Fallback: Use spatial filtering only"""
       matched_uuids = set()
       
       for perc_id, perc_data in perception_data.items():
           pos = perc_data['pose'].position
           dist_from_stack = np.sqrt((pos.x - Config.BASE_X)**2 + (pos.y - Config.BASE_Y)**2)
           
           # Find existing record
           found = False
           for uuid, record in self.records.items():
               if record.state == "REMOVED":
                   continue
               if record.current_perception_id == perc_id:
                   record.update(perc_data, 0.0)
                   matched_uuids.add(uuid)
                   found = True
                   break
           
           if not found:
               new_record = CubeRecord(perc_data)
               if dist_from_stack < Config.STACK_VERIFICATION_RADIUS:
                   new_record.state = "PLACED"
               self.records[new_record.uuid] = new_record
               matched_uuids.add(new_record.uuid)
       
       # Mark misses
       for uuid, record in self.records.items():
           if uuid not in matched_uuids and record.state != "REMOVED":
               record.mark_miss()
   
   def _simple_update(self, perception_data):
       """Simple fallback"""
       for perc_id, perc_data in perception_data.items():
           new_record = CubeRecord(perc_data)
           self.records[new_record.uuid] = new_record
   
   def _detect_removals(self):
       """Detect and log cube removals"""
       newly_removed = []
       
       for uuid, record in self.records.items():
           if record.state == "REMOVED":
               continue
           
           # Check if cube should be marked as removed
           if record.mark_miss():
               newly_removed.append((uuid, record))
               self.removed_count += 1
       
       if newly_removed:
           rospy.logwarn(f"[TrackingManager] 🗑️  Detected {len(newly_removed)} removed cube(s):")
           for uuid, record in newly_removed:
               rospy.logwarn(f"    - UUID {uuid}: was {record.state}, "
                           f"last seen {(rospy.Time.now() - record.last_seen).to_sec():.1f}s ago")
   
   def _log_status(self):
       """Log current tracking status"""
       available = len(self.get_available_cubes())
       picked = len([r for r in self.records.values() if r.state == "PICKED"])
       placed = len([r for r in self.records.values() if r.state == "PLACED"])
       removed = len([r for r in self.records.values() if r.state == "REMOVED"])
       
       rospy.loginfo(f"[TrackingManager] Status: {available} available, {picked} picked, "
                    f"{placed} placed, {removed} removed (total tracked: {len(self.records)})")
   
   def get_available_cubes(self):
       """Return cubes available for picking"""
       return {uuid: rec for uuid, rec in self.records.items() 
               if rec.state == "AVAILABLE" and 
               rec.tracking_confidence > Config.LOW_CONFIDENCE_THRESHOLD}
   
   def get_cube(self, uuid):
       return self.records.get(uuid)
   
   def mark_picked(self, uuid):
       if uuid in self.records:
           self.records[uuid].mark_picked()
   
   def mark_placed(self, uuid, expected_position):
       if uuid in self.records:
           self.records[uuid].mark_placed(expected_position)
           self.placed_count += 1
   
   def verify_placement(self, uuid, target_z):
       """Verify cube was placed correctly"""
       if uuid not in self.records:
           rospy.loginfo(f"[VerifyPlacement] ✅ UUID {uuid} not found - "
                        f"assuming successfully placed and renamed")
           return True
       
       record = self.records[uuid]
       
       # If cube was removed, placement failed
       if record.state == "REMOVED":
           rospy.logwarn(f"[VerifyPlacement] ❌ UUID {uuid} was removed from scene")
           return False
       
       # Check if cube is at expected position
       if record.verify_placement(target_z):
           return True
       
       # Fallback: Check cube counts
       available_count = len(self.get_available_cubes())
       expected_remaining = self.initial_cube_count - self.placed_count - self.removed_count
       
       if available_count == expected_remaining:
           rospy.loginfo(f"[VerifyPlacement] ✅ Cube count matches "
                        f"({available_count} available, {self.placed_count} placed, "
                        f"{self.removed_count} removed)")
           record.placement_verified = True
           return True
       
       rospy.logwarn(f"[VerifyPlacement] ❌ Count mismatch: expected {expected_remaining}, "
                    f"found {available_count}")
       return False
   
   def handle_task_failure(self):
       """Handle failed pick/place by resetting states if needed"""
       # Reset any PICKED cubes back to AVAILABLE if task failed
       for uuid, record in self.records.items():
           if record.state == "PICKED":
               rospy.logwarn(f"[TrackingManager] Resetting UUID {uuid} from PICKED to AVAILABLE")
               record.state = "AVAILABLE"
               record.consecutive_misses = 0

class BB:
   TRACKING_MANAGER = "tracking_manager"
   GOAL_IDX = "goal_idx"
   SELECTED_UUID = "selected_uuid"
   TARGET_POSE = "target"
   TRAJECTORY = "traj"

def init_blackboard():
   bb = py_trees.blackboard.Blackboard()
   bb.set(BB.TRACKING_MANAGER, CubeTrackingManager())
   bb.set(BB.GOAL_IDX, 0)
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
               # Reset picked cubes on failure
               bb = py_trees.blackboard.Blackboard()
               manager = bb.get(BB.TRACKING_MANAGER)
               manager.handle_task_failure()
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
           rospy.logwarn("[SelectCube] No available cubes!")
           return py_trees.common.Status.FAILURE
       
       # Select cube farthest from stack (highest X)
       best_uuid = max(available.keys(), 
                      key=lambda u: available[u].current_pose.position.x)
       
       cube = available[best_uuid]
       self.bb.set(BB.SELECTED_UUID, best_uuid)
       rospy.loginfo(f"[SelectCube] ✅ Selected UUID {best_uuid} "
                    f"(cube_{cube.current_perception_id}, conf={cube.tracking_confidence:.2f})")
       return py_trees.common.Status.SUCCESS

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
           rospy.loginfo(f"[MarkPicked] UUID {uuid} marked as PICKED")
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
           idx = self.bb.get(BB.GOAL_IDX) or 0
           self.bb.set(BB.GOAL_IDX, idx + 1)
           rospy.loginfo(f"[MarkPlaced] UUID {uuid} marked as PLACED (stack height: {idx + 1})")
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
           if not cube or cube.state == "REMOVED":
               rospy.logerr(f"[PlanToPose] Cube UUID {uuid} not available (removed or not found)")
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
       
       if not cube or cube.state == "REMOVED":
           rospy.logerr(f"[GripperClose] Cube UUID {uuid} not available")
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
       
       if not cube or cube.state == "REMOVED":
           rospy.logerr(f"[Attach] Cube UUID {uuid} not available")
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
   
   # Verification with removal detection
   verification = py_trees.composites.Sequence("Verification")
   verification.add_child(Perceive())
   verification.add_child(VerifyPlacement())
   loop.add_child(Retry("RetryVerification", verification, 2))
   
   root.add_child(loop)
   return root

def main():
   rospy.init_node('orchestrator_bt_v6_improved')
   init_blackboard()
   tree = py_trees_ros.trees.BehaviourTree(build_tree())
   if tree.setup(timeout=30):
       rospy.loginfo("🚀 Improved Removal Detection Tree setup successful. Starting...")
       tree.tick_tock(100)
   else:
       rospy.logerr("Tree setup failed.")

if __name__ == '__main__':
   main()
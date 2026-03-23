#!/usr/bin/env python3
"""
VERSION 6: BB-Authoritative Perception
=======================================
PERCEPTION RULES:
  - First run (BB empty): register ALL perceived cubes as new CubeRecords.
  - Subsequent runs: BB is ground truth.
      * Each BB cube is matched to its nearest perceived cube (by XY distance).
      * If a match is found within POSITION_MATCH_THRESHOLD:
            → update the BB cube's pose and current_perception_id.
      * If NO match is found for a BB cube:
            → remove it from BB (it physically disappeared).
      * Perceived cubes that are not matched to any BB cube → AVAILABLE (as a new cube)
        (prevents ghost cubes / ID drift from renaming).
  - PLACED cubes are re-matched so their pose stays current but they are
    never demoted back to AVAILABLE.
"""

from scipy.spatial.transform import Rotation as R
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

    # Matching threshold (metres) – BB cube is removed if no perception within this radius
    POSITION_MATCH_THRESHOLD = 0.06
    STACK_VERIFICATION_RADIUS = 0.08
    STACK_HEIGHT_TOLERANCE = 0.03

    HIGH_CONFIDENCE_THRESHOLD = 0.85
    LOW_CONFIDENCE_THRESHOLD = 0.50

    SAFE_HEIGHT = 0.2
    GRASP_HEIGHT = 0.0
    PLACE_HEIGHT = 0.0015

    GRIPPER_OPEN_WIDTH = 0.08
    GRIPPER_FORCE = 30.0
    GRIPPER_WIDTH_MARGIN = 0.005
    MIN_GRIPPER_WIDTH = 0.01
    GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]

    MAX_PICK_RETRIES = 2
    MAX_PLACE_RETRIES = 3


# ---------------------------------------------------------------------------
# CubeRecord  (unchanged from v5 except minor cleanup)
# ---------------------------------------------------------------------------

class CubeRecord:
    """Tracks a single cube across perception frames."""

    def __init__(self, initial_perception_data):
        self.uuid = self._generate_uuid()
        self.initial_pose = deepcopy(initial_perception_data['pose'])
        self.current_pose = deepcopy(initial_perception_data['pose'])
        self.dimensions = initial_perception_data['dimensions']
        self.current_perception_id = initial_perception_data['id']

        self.state = "AVAILABLE"        # AVAILABLE | PICKED | PLACED | LOST
        self.tracking_confidence = 1.0
        self.last_seen = rospy.Time.now()
        self.match_count = 1
        self.total_distance_moved = 0.0

        self.placement_verified = False
        self.expected_place_position = None

    @staticmethod
    def _generate_uuid():
        import uuid
        return str(uuid.uuid4())[:8]

    def xy_distance_to(self, perc_data):
        """2-D (XY) distance between this record and a perception entry."""
        px = perc_data['pose'].position.x
        py = perc_data['pose'].position.y
        dx = self.current_pose.position.x - px
        dy = self.current_pose.position.y - py
        return np.sqrt(dx * dx + dy * dy)

    def update_from_perception(self, perc_data):
        """Overwrite pose and perception ID; keep state / uuid."""
        old_pos = np.array([self.current_pose.position.x,
                             self.current_pose.position.y,
                             self.current_pose.position.z])

        self.current_pose = deepcopy(perc_data['pose'])
        self.current_perception_id = perc_data['id']
        self.last_seen = rospy.Time.now()
        self.match_count += 1

        new_pos = np.array([self.current_pose.position.x,
                             self.current_pose.position.y,
                             self.current_pose.position.z])
        self.total_distance_moved += np.linalg.norm(new_pos - old_pos)
        self.tracking_confidence = 1.0          # fresh match → full confidence

    # ---- state helpers ------------------------------------------------
    def is_on_stack(self):
        dist = np.sqrt((self.current_pose.position.x - Config.BASE_X) ** 2 +
                       (self.current_pose.position.y - Config.BASE_Y) ** 2)
        return dist < Config.STACK_VERIFICATION_RADIUS

    def mark_picked(self):
        self.state = "PICKED"

    def mark_placed(self, expected_position):
        self.state = "PLACED"
        self.placement_verified = False
        self.expected_place_position = expected_position

    def verify_placement(self, target_z):
        z_error = abs(self.current_pose.position.z - target_z)
        if z_error < Config.STACK_HEIGHT_TOLERANCE and self.is_on_stack():
            self.placement_verified = True
            return True
        return False


# ---------------------------------------------------------------------------
# CubeTrackingManager
# ---------------------------------------------------------------------------

class CubeTrackingManager:
    """
    BB-authoritative cube tracker.

    update() behaviour depends on whether any records exist:
      • BB empty  → register everything from perception (first call).
      • BB has records → match each BB cube to nearest perceived cube;
                         remove BB cubes that have no match.
                         Do NOT add unmatched perceived cubes.
    """

    def __init__(self):
        self.records = {}           # uuid → CubeRecord
        self.placed_count = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(self, perception_data: dict):
        """
        perception_data: {int_id: {'id': int, 'pose': Pose, 'dimensions': Vector3}}
        """
        n_perceived = len(perception_data)
        n_bb = len(self.records)
        rospy.loginfo(f"[TrackingManager] update: {n_perceived} perceived, {n_bb} in BB")

        if n_bb == 0:
            # ── First call: register everything ──────────────────────────
            rospy.loginfo("[TrackingManager] BB empty → registering all perceived cubes")
            for perc_id, perc_data in perception_data.items():
                rec = CubeRecord(perc_data)
                self.records[rec.uuid] = rec
                rospy.loginfo(f"  Registered UUID {rec.uuid} ← perception id {perc_id} "
                              f"@ ({perc_data['pose'].position.x:.3f}, "
                              f"{perc_data['pose'].position.y:.3f}, "
                              f"{perc_data['pose'].position.z:.3f})")
            return

        # ── Subsequent calls: BB is ground truth ─────────────────────────
        perc_list = list(perception_data.values())   # list of perc dicts
        used_perc_indices = set()                     # which perc entries got matched

        uuids_to_remove = []

        for uuid, record in self.records.items():

            # PICKED cubes are in the gripper – skip matching, keep record
            if record.state == "PICKED":
                rospy.loginfo(f"  UUID {uuid}: PICKED – skip matching")
                continue

            # Find nearest perceived cube (XY distance)
            best_idx = None
            best_dist = float('inf')
            for i, perc_data in enumerate(perc_list):
                if i in used_perc_indices:
                    continue
                dist = record.xy_distance_to(perc_data)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None and best_dist < Config.POSITION_MATCH_THRESHOLD:
                # ✅ Match found → update pose and perception ID
                matched = perc_list[best_idx]
                old_pid = record.current_perception_id
                record.update_from_perception(matched)
                used_perc_indices.add(best_idx)
                rospy.loginfo(
                    f"  UUID {uuid} ({record.state}): matched perception id "
                    f"{old_pid} → {matched['id']} (dist={best_dist:.4f} m)"
                )
            else:
                # ❌ No match within threshold
                if record.state == "PLACED":
                    # Placed cubes can disappear (renamed in Gazebo) – keep but warn
                    rospy.logwarn(
                        f"  UUID {uuid} (PLACED): no perception match "
                        f"(best dist={best_dist:.4f} m) – keeping record"
                    )
                else:
                    # AVAILABLE cube vanished → remove from BB
                    rospy.logwarn(
                        f"  UUID {uuid} ({record.state}): no perception match "
                        f"(best dist={best_dist:.4f} m) – REMOVING from BB"
                    )
                    uuids_to_remove.append(uuid)

        for uuid in uuids_to_remove:
            del self.records[uuid]

        # Add unmatched perceived cubes as new AVAILABLE records
        for i, perc_data in enumerate(perc_list):
            if i in used_perc_indices:
                continue
            new_rec = CubeRecord(perc_data)
            self.records[new_rec.uuid] = new_rec
            rospy.loginfo(
                f"  New cube: UUID {new_rec.uuid} ← perception id {perc_data['id']} "
                f"@ ({perc_data['pose'].position.x:.3f}, "
                f"{perc_data['pose'].position.y:.3f}, "
                f"{perc_data['pose'].position.z:.3f}) – added as AVAILABLE"
            )

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def get_available_cubes(self):
        return {uuid: rec for uuid, rec in self.records.items()
                if rec.state == "AVAILABLE"}

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
        """Returns True if placement confirmed; also handles renamed/disappeared cubes."""
        if uuid not in self.records:
            rospy.loginfo(
                f"[VerifyPlacement] UUID {uuid} not in BB – "
                f"assuming renamed/placed successfully"
            )
            return True

        record = self.records[uuid]
        if record.verify_placement(target_z):
            return True

        # Fallback: cube count bookkeeping
        available_count = len(self.get_available_cubes())
        rospy.logwarn(
            f"[VerifyPlacement] UUID {uuid} spatial check failed – "
            f"{available_count} AVAILABLE cubes remain"
        )
        return False


# ---------------------------------------------------------------------------
# Blackboard keys
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RETRY DECORATOR  (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BEHAVIORS
# ---------------------------------------------------------------------------

class Perceive(py_trees.behaviour.Behaviour):
    """
    Calls the perception service and updates the CubeTrackingManager.

    BB-authoritative logic lives entirely in CubeTrackingManager.update().
    This node just bridges ROS ↔ the manager.
    """

    def __init__(self):
        super(Perceive, self).__init__("Perceive")
        self.bb = py_trees.blackboard.Blackboard()
        self.proxy = None

    def setup(self, timeout):
        try:
            rospy.wait_for_service('/perception_service', timeout=timeout)
            self.proxy = rospy.ServiceProxy('/perception_service', PerceptionService)
            return True
        except Exception as e:
            rospy.logerr(f"[Perceive] setup failed: {e}")
            return False

    def update(self):
        try:
            resp = self.proxy(trigger=True)
            if not resp.success:
                rospy.logwarn("[Perceive] Perception service returned failure")
                return py_trees.common.Status.FAILURE

            # Build perception dict keyed by index
            perceived = {}
            for i in range(resp.num_cubes):
                perceived[i] = {
                    'id': i,
                    'pose': resp.cube_poses.poses[i],
                    'dimensions': resp.dimensions[i],
                }

            rospy.loginfo(f"[Perceive] Service returned {len(perceived)} cube(s)")

            manager: CubeTrackingManager = self.bb.get(BB.TRACKING_MANAGER)
            manager.update(perceived)

            # Summary log
            available = manager.get_available_cubes()
            all_recs = manager.records
            rospy.loginfo(
                f"[Perceive] BB after update: {len(all_recs)} total records, "
                f"{len(available)} AVAILABLE"
            )
            for uuid, rec in all_recs.items():
                rospy.loginfo(
                    f"  UUID {uuid} | perc_id={rec.current_perception_id} "
                    f"| state={rec.state} "
                    f"| pos=({rec.current_pose.position.x:.3f}, "
                    f"{rec.current_pose.position.y:.3f}, "
                    f"{rec.current_pose.position.z:.3f})"
                )

            return py_trees.common.Status.SUCCESS

        except Exception as e:
            rospy.logerr(f"[Perceive] Error: {e}")
            import traceback; traceback.print_exc()
            return py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# Remaining behaviors — identical to v5, reproduced here for completeness
# ---------------------------------------------------------------------------

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

        best_uuid = max(available.keys(),
                        key=lambda u: available[u].current_pose.position.x)

        cube = available[best_uuid]
        self.bb.set(BB.SELECTED_UUID, best_uuid)
        rospy.loginfo(
            f"[SelectCube] ✅ Selected UUID {best_uuid} "
            f"(perc_id={cube.current_perception_id}, "
            f"conf={cube.tracking_confidence:.2f})"
        )
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
            rospy.loginfo(
                f"[MarkPlaced] UUID {uuid} marked as PLACED "
                f"(GOAL_IDX updated by ReconcileStack)"
            )
        return py_trees.common.Status.SUCCESS


class ReconcileStack(py_trees.behaviour.Behaviour):
    """
    Re-perceives after placement, counts cubes on stack, corrects GOAL_IDX,
    and syncs MoveIt collision objects.  Always returns SUCCESS.
    """

    def __init__(self):
        super(ReconcileStack, self).__init__("ReconcileStack")
        self.bb = py_trees.blackboard.Blackboard()
        self.proxy = None
        self._scene = None

    def setup(self, timeout):
        try:
            rospy.wait_for_service('/perception_service', timeout=timeout)
            self.proxy = rospy.ServiceProxy('/perception_service', PerceptionService)
            import moveit_commander
            self._scene = moveit_commander.PlanningSceneInterface()
            return True
        except Exception as e:
            rospy.logerr(f"[ReconcileStack] setup failed: {e}")
            return False

    def _is_on_stack(self, pose):
        dx = pose.position.x - Config.BASE_X
        dy = pose.position.y - Config.BASE_Y
        return (dx * dx + dy * dy) ** 0.5 < Config.STACK_VERIFICATION_RADIUS

    def _expected_stack_z(self, layer):
        return Config.BASE_Z + layer * Config.CUBE_SIZE + Config.CUBE_SIZE / 2.0

    def _remove_stale_placed_collisions(self, valid_count):
        try:
            for name in self._scene.get_known_object_names():
                if not name.startswith("placed_cube_"):
                    continue
                try:
                    idx = int(name.split("placed_cube_")[1])
                except ValueError:
                    continue
                if idx >= valid_count:
                    self._scene.remove_world_object(name)
                    rospy.logwarn(
                        f"[ReconcileStack] Removed stale '{name}' "
                        f"(stack height now {valid_count})"
                    )
        except Exception as e:
            rospy.logerr(f"[ReconcileStack] collision cleanup error: {e}")

    def _add_or_update_stack_collisions(self, stack_cubes):
        from geometry_msgs.msg import PoseStamped as PS
        sorted_cubes = sorted(stack_cubes, key=lambda d: d['pose'].position.z)
        for layer, data in enumerate(sorted_cubes):
            name = f"placed_cube_{layer}"
            ps = PS()
            ps.header.frame_id = "world"
            ps.header.stamp = rospy.Time.now()
            ps.pose = data['pose']
            ps.pose.position.z = self._expected_stack_z(layer)
            self._scene.add_box(
                name, ps,
                size=(Config.CUBE_SIZE, Config.CUBE_SIZE, Config.CUBE_SIZE)
            )

    def update(self):
        rospy.loginfo("[ReconcileStack] Starting post-placement reconciliation …")

        try:
            resp = self.proxy(trigger=True)
        except Exception as e:
            rospy.logerr(f"[ReconcileStack] Perception call failed: {e}")
            return py_trees.common.Status.SUCCESS

        if not resp.success:
            rospy.logwarn("[ReconcileStack] Perception returned failure, skipping")
            return py_trees.common.Status.SUCCESS

        detected = [{'id': i,
                     'pose': resp.cube_poses.poses[i],
                     'dimensions': resp.dimensions[i]}
                    for i in range(resp.num_cubes)]

        on_stack = [d for d in detected if self._is_on_stack(d['pose'])]
        confirmed_height = len(on_stack)
        rospy.loginfo(
            f"[ReconcileStack] {len(detected)} total detected, "
            f"{confirmed_height} on stack"
        )

        old_idx = self.bb.get(BB.GOAL_IDX) or 0
        self.bb.set(BB.GOAL_IDX, confirmed_height)
        if confirmed_height != old_idx:
            rospy.logwarn(
                f"[ReconcileStack] GOAL_IDX: {old_idx} → {confirmed_height}"
            )

        manager: CubeTrackingManager = self.bb.get(BB.TRACKING_MANAGER)
        manager.placed_count = confirmed_height

        self._remove_stale_placed_collisions(confirmed_height)
        if on_stack:
            self._add_or_update_stack_collisions(on_stack)

        # Update BB with fresh perception (BB-authoritative matching)
        perception_dict = {d['id']: d for d in detected}
        manager.update(perception_dict)

        rospy.loginfo(
            f"[ReconcileStack] ✅ Done. height={confirmed_height}, "
            f"next Z={self._expected_stack_z(confirmed_height):.4f} m"
        )
        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    if quat is not None:
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
    return p


# ---------------------------------------------------------------------------
# Action client base
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Specific action behaviors  (unchanged from v5)
# ---------------------------------------------------------------------------

class PlanHome(ActionClient):
    def __init__(self):
        super().__init__("PlanHome", "/planning_action", PlanningActionAction)

    def make_goal(self):
        g = PlanningActionGoal()
        g.action = "HOME"
        return g

    def on_success(self, res):
        self.bb.set(BB.TRAJECTORY, res.trajectory)
        return py_trees.common.Status.SUCCESS


class PlanToPose(ActionClient):
    def __init__(self, name, pose_source, offset, col=False, mode=""):
        super().__init__(name, "/planning_action", PlanningActionAction)
        self.pose_source = pose_source
        self.offset = offset
        self.col = col
        self.mode = mode

    def make_goal(self):
        g = PlanningActionGoal()
        g.action = self.mode

        if self.pose_source == "SELECTED":
            manager = self.bb.get(BB.TRACKING_MANAGER)
            uuid = self.bb.get(BB.SELECTED_UUID)
            cube = manager.get_cube(uuid)
            if not cube:
                return None
            base = cube.current_pose
            cube_rot = R.from_quat([base.orientation.x, base.orientation.y,
                                    base.orientation.z, base.orientation.w])
            gripper_down = R.from_euler('x', 180, degrees=True)
            final_quat = (cube_rot * gripper_down).as_quat()
            g.target_pose = offset_pose(base, *self.offset, quat=final_quat)
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
        super().__init__("Execute", "/control_action", ControlActionAction)

    def make_goal(self):
        traj = self.bb.get(BB.TRAJECTORY)
        g = ControlActionGoal()
        g.trajectory = traj
        return g if traj else None

    def on_success(self, res):
        return py_trees.common.Status.SUCCESS


class GripperClose(ActionClient):
    def __init__(self):
        super().__init__("GripperClose", "/gripper_action", GripperActionAction)

    def make_goal(self):
        manager = self.bb.get(BB.TRACKING_MANAGER)
        uuid = self.bb.get(BB.SELECTED_UUID)
        cube = manager.get_cube(uuid)
        d = cube.dimensions
        width = min(d.x, d.y) if hasattr(d, 'x') else min(d[0], d[1])
        g = GripperActionGoal()
        g.open = False
        g.width = max(Config.MIN_GRIPPER_WIDTH, width - Config.GRIPPER_WIDTH_MARGIN)
        g.force = Config.GRIPPER_FORCE
        return g

    def on_success(self, res):
        return (py_trees.common.Status.SUCCESS
                if res and res.final_width > 0.005
                else py_trees.common.Status.FAILURE)


class GripperOpen(ActionClient):
    def __init__(self):
        super().__init__("GripperOpen", "/gripper_action", GripperActionAction)

    def make_goal(self):
        g = GripperActionGoal()
        g.open = True
        g.width = Config.GRIPPER_OPEN_WIDTH
        return g

    def on_success(self, res):
        return py_trees.common.Status.SUCCESS


class Attach(ActionClient):
    def __init__(self):
        super().__init__("Attach", "/planning_action", PlanningActionAction)

    def make_goal(self):
        manager = self.bb.get(BB.TRACKING_MANAGER)
        uuid = self.bb.get(BB.SELECTED_UUID)
        cube = manager.get_cube(uuid)
        g = PlanningActionGoal()
        g.action = "ATTACH"
        g.object_name = f"cube_{cube.current_perception_id}"
        return g

    def on_success(self, res):
        return py_trees.common.Status.SUCCESS


class Detach(ActionClient):
    def __init__(self):
        super().__init__("Detach", "/planning_action", PlanningActionAction)

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


# ---------------------------------------------------------------------------
# TREE BUILDING  (unchanged from v5)
# ---------------------------------------------------------------------------

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
    pick.add_child(PlanToPose("ApproachPick", "SELECTED", (0, 0, Config.SAFE_HEIGHT)))
    pick.add_child(Execute())
    pick.add_child(PlanToPose("DescendPick", "SELECTED", (0, 0, Config.GRASP_HEIGHT)))
    pick.add_child(Execute())
    pick.add_child(GripperClose())
    pick.add_child(Attach())
    pick.add_child(MarkPicked())
    pick.add_child(PlanToPose("Lift", "SELECTED", (0, 0, Config.SAFE_HEIGHT)))
    pick.add_child(Execute())
    loop.add_child(Retry("RetryPick", pick, Config.MAX_PICK_RETRIES))

    place = py_trees.composites.Sequence("PlaceSequence")
    place.add_child(PlanToPose("ApproachPlace", BB.TARGET_POSE, (0, 0, Config.SAFE_HEIGHT), True))
    place.add_child(Execute())
    place.add_child(PlanToPose("DescendPlace", BB.TARGET_POSE, (0, 0, Config.PLACE_HEIGHT), True, mode="STACK"))
    place.add_child(Execute())
    place.add_child(Detach())
    place.add_child(GripperOpen())
    place.add_child(MarkPlaced())
    place.add_child(PlanToPose("Clear", BB.TARGET_POSE, (0, 0, Config.SAFE_HEIGHT), True, mode="STACK"))
    place.add_child(Execute())
    place.add_child(ReconcileStack())
    loop.add_child(Retry("RetryPlace", place, Config.MAX_PLACE_RETRIES))

    root.add_child(loop)
    return root


def main():
    rospy.init_node('orchestrator_bt_v6_perception_fixed')
    init_blackboard()
    tree = py_trees_ros.trees.BehaviourTree(build_tree())
    if tree.setup(timeout=30):
        rospy.loginfo("🚀 BB-Authoritative Tracking Tree ready. Starting…")
        tree.tick_tock(100)
    else:
        rospy.logerr("Tree setup failed.")


if __name__ == '__main__':
    main()
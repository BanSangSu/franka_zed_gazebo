#!/usr/bin/env python3


import rospy
import moveit_commander
import actionlib
from threading import Lock
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped


from franka_zed_gazebo.msg import (
   PlanningActionAction, PlanningActionFeedback, PlanningActionResult, PlanningActionGoal
)




class PlanningActionServer:
   """
   Enhanced Motion Planning Action Server for Franka Panda with MoveIt.
  
   Provides motion planning, object attachment/detachment, and home positioning
   with proper feedback, preemption support, and collision management.
   """
  
   def __init__(self):
       rospy.init_node('planning_action_server')
      
       # Thread safety for MoveIt operations
       self.planning_lock = Lock()
      
       # Initialize moveit_commander
       moveit_commander.roscpp_initialize([])
      
       self.robot = moveit_commander.RobotCommander()
       self.scene = moveit_commander.PlanningSceneInterface()
      
       # Store object metadata for re-adding after detachment
       self.attached_objects = {}  # object_name -> {size, original_pose, attach_time}

    #    self.cube_size = (0.040, 0.040, 0.040)
       self.cube_size = (0.035, 0.035, 0.035)
      
       # Load parameters with validation
       self.planning_group = rospy.get_param('~planning_group', 'panda_manipulator')
       self.ee_link = rospy.get_param('~ee_link', 'panda_hand_tcp')
      
       # Validate group exists
       if self.planning_group not in self.robot.get_group_names():
           rospy.logfatal(f"Planning group '{self.planning_group}' not found. Available: {self.robot.get_group_names()}")
           raise ValueError(f"Invalid planning group: {self.planning_group}")
      
       self.move_group = moveit_commander.MoveGroupCommander(self.planning_group)
      
       # Configure planner with validated parameters
       planner_id = rospy.get_param('~planner_id', 'RRTConnect')
       planning_time = rospy.get_param('~planning_time', 20.0)
       num_attempts = rospy.get_param('~num_attempts', 30)
      
       self.move_group.set_planner_id(planner_id)
       self.move_group.set_planning_time(planning_time)
       self.move_group.set_num_planning_attempts(num_attempts)
       self.move_group.allow_replanning(True)
      
       # Set velocity and acceleration scaling
       max_velocity = rospy.get_param('~max_velocity', 0.20) # 0.15
       max_acceleration = rospy.get_param('~max_acceleration', 0.12) # 0.10
       self.move_group.set_max_velocity_scaling_factor(max_velocity)
       self.move_group.set_max_acceleration_scaling_factor(max_acceleration)
      
       # Goal tolerance for better planning success
       self.move_group.set_goal_position_tolerance(rospy.get_param('~goal_position_tolerance', 0.005))
       self.move_group.set_goal_orientation_tolerance(rospy.get_param('~goal_orientation_tolerance', 0.01))
      
       # Touch links for attachment
       self.touch_links = [self.ee_link, "panda_leftfinger", "panda_rightfinger", "panda_hand"]
      
       # Service proxies for planning scene manipulation
       get_planning_scene= rospy.resolve_name('get_planning_scene')
       apply_planning_scene = rospy.resolve_name('apply_planning_scene')
       rospy.wait_for_service(get_planning_scene, timeout=10.0)
       rospy.wait_for_service(apply_planning_scene, timeout=10.0)
       self.get_scene_srv = rospy.ServiceProxy(get_planning_scene, GetPlanningScene)
       self.apply_scene_srv = rospy.ServiceProxy(apply_planning_scene, ApplyPlanningScene)
      
       # Create action server with preemption support
       self.server = actionlib.SimpleActionServer(
           '/planning_action',
           PlanningActionAction,
           execute_cb=self.execute_cb,
           auto_start=False
       )
       self.server.register_preempt_callback(self.preempt_cb)
       self.server.start()
      
       rospy.loginfo(f"â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
       rospy.loginfo(f"Planning Action Server Ready")
       rospy.loginfo(f"  Group: {self.planning_group}")
       rospy.loginfo(f"  End Effector: {self.ee_link}")
       rospy.loginfo(f"  Planner: {planner_id}")
       rospy.loginfo(f"  Planning Time: {planning_time}s")
       rospy.loginfo(f"  Max Velocity: {max_velocity}")
       rospy.loginfo(f"  Max Acceleration: {max_acceleration}")
       rospy.loginfo(f"â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
  
  
   def preempt_cb(self):
       """Handle preemption requests gracefully"""
       rospy.logwarn("Planning preempted by client")
       with self.planning_lock:
           self.move_group.stop()
           self.move_group.clear_pose_targets()
      
       result = PlanningActionResult()
       result.success = False
       result.message = "Planning preempted by client request"
       self.server.set_preempted(result)
  
  
   def publish_feedback(self, progress, status):
       """Publish feedback with preemption check"""
       if self.server.is_preempt_requested():
           return False
      
       feedback = PlanningActionFeedback()
       feedback.progress = progress
       feedback.status = status
       self.server.publish_feedback(feedback)
       return True
  
  
   def get_object_from_scene(self, object_name):
       """Retrieve object from planning scene with error handling"""
       try:
           objects = self.scene.get_objects([object_name])
           if object_name in objects:
               return objects[object_name]
           rospy.logwarn(f"Object '{object_name}' not found in planning scene")
           return None
       except Exception as e:
           rospy.logerr(f"Error retrieving object '{object_name}': {e}")
           return None
  
  
   def handle_attach(self, goal):
       """Handle object attachment with verification"""
       result = PlanningActionResult()
      
       if not goal.object_name:
           result.success = False
           result.message = "ATTACH action requires object_name"
           return result
      
       if not self.publish_feedback(0.2, f"Retrieving {goal.object_name} from scene"):
           return None  # Preempted
      
       # Get object dimensions before attaching
       size = self.cube_size  # Default size
       obj = self.get_object_from_scene(goal.object_name)
       if obj is None:
           rospy.logwarn(f"Cannot find {goal.object_name} in scene, attempting attach anyway")
       else:
           if obj.primitives and len(obj.primitives) > 0:
               size = tuple(obj.primitives[0].dimensions)
           else:
               rospy.logwarn(f"No primitive shape for {goal.object_name}, using default size")
      
       # Store metadata
       current_pose = self.move_group.get_current_pose(self.ee_link)
       self.attached_objects[goal.object_name] = {
           'size': self.cube_size,
           'attach_pose': current_pose,
           'attach_time': rospy.Time.now()
       }
      
       if not self.publish_feedback(0.5, f"Attaching {goal.object_name}"):
           return None
      
       # Attach with touch links
       self.scene.attach_box(
           self.ee_link,
           goal.object_name,
           touch_links=self.touch_links
       )
      
       rospy.sleep(0.5)  # Allow planning scene to update
      
       # Verify attachment
       attached_objects = self.scene.get_attached_objects([goal.object_name])
       if goal.object_name in attached_objects:
           rospy.loginfo(f"âœ“ Attached {goal.object_name} to {self.ee_link} (size: {size})")
          
           if not self.publish_feedback(1.0, "Attach complete"):
               return None
          
           result.success = True
           result.message = f"Object {goal.object_name} attached successfully"
       else:
           rospy.logwarn(f"âœ— Attachment verification failed for {goal.object_name}")
           result.success = False
           result.message = f"Attachment verification failed"
      
       return result
  
  
   def handle_detach(self, goal):
       """Handle object detachment with scene update"""
       result = PlanningActionResult()
      
       # Determine if we are detaching a specific object or all
       target_name = goal.object_name
       detach_all = not target_name or target_name.lower() == "all"
      
       if not self.publish_feedback(0.3, f"Detaching {'all objects' if detach_all else target_name}"):
           return None
      
       # Get current pose before detaching
       current_pose = self.move_group.get_current_pose(self.ee_link)
      
       if detach_all:
           # Detach everything from the end effector link
           try:
               # remove_attached_object(link, name=None) removes all if name is None
               self.scene.remove_attached_object(self.ee_link)
               rospy.loginfo(f"âœ“ Detached all objects from {self.ee_link}")
               result.success = True
               result.message = "All objects detached successfully"
           except Exception as e:
               rospy.logerr(f"Error during detach all: {e}")
               result.success = False
               result.message = f"Error during detach all: {str(e)}"
           return result


       # --- Specific Object Detach Logic ---
       # Retrieve stored metadata
       if target_name not in self.attached_objects:
           rospy.logwarn(f"No metadata for {target_name}, using defaults")
           size = (0.045, 0.045, 0.045)
       else:
           metadata = self.attached_objects[target_name]
           size = metadata['size']
           rospy.loginfo(f"Detaching {target_name} after {(rospy.Time.now() - metadata['attach_time']).to_sec():.1f}s")
      
       # Detach specific object
       self.scene.remove_attached_object(self.ee_link, name=target_name)
       rospy.sleep(0.5)  # Allow scene update
      
       if not self.publish_feedback(0.6, "Adding object back to world"):
           return None
      
       # Generate new name to prevent perception overwrite
       new_name = target_name.replace("cube_", "placed_cube_")
      
       # Add back to world scene at current location
       self.scene.add_box(new_name, current_pose, size)
       rospy.sleep(0.3)
      
       # Remove old world object if name changed
       if new_name != target_name:
           self.scene.remove_world_object(target_name)
      
       # Clean up metadata
       if target_name in self.attached_objects:
           del self.attached_objects[target_name]
      
       rospy.loginfo(f"âœ“ Detached {target_name} â†’ {new_name} at [{current_pose.pose.position.x:.3f}, {current_pose.pose.position.y:.3f}, {current_pose.pose.position.z:.3f}]")
      
       if not self.publish_feedback(1.0, "Detach complete"):
           return None
      
       result.success = True
       result.message = f"Object {target_name} detached and added as {new_name}"
       return result
  
  
   def handle_home(self, goal):
       """Move to home/ready position"""
       result = PlanningActionResult()
      
       if not self.publish_feedback(0.2, "Planning to Home (Ready) pose"):
           return None
      
       with self.planning_lock:
           try:
               # Try named target first
               self.move_group.set_named_target("ready")
               rospy.loginfo("Using 'ready' named target")
           except:
               # Fallback to joint values
               rospy.logwarn("Named pose 'ready' not found, using joint fallback")
               home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
               self.move_group.set_joint_value_target(home_joints)
          
           if not self.publish_feedback(0.5, "Computing home trajectory"):
               return None
          
           plan = self.move_group.plan()
          
           if isinstance(plan, tuple):
               success, trajectory, _, _ = plan
           else:
               trajectory = plan
               success = len(trajectory.joint_trajectory.points) > 0
          
           self.move_group.clear_pose_targets()
      
       if success:
           if not self.publish_feedback(1.0, "Home planning succeeded"):
               return None
          
           result.success = True
           result.message = "Home planning succeeded"
           result.trajectory = trajectory
           rospy.loginfo("âœ“ Home planning complete")
       else:
           result.success = False
           result.message = "Home planning failed"
           rospy.logwarn("âœ— Home planning failed")
      
       return result
  
  
   def handle_motion_planning(self, goal):
       """Handle motion planning with optional collision allowances"""
       result = PlanningActionResult()
      
       if not self.publish_feedback(0.1, "Validating target pose"):
           return None
      
       with self.planning_lock:
           # Set pose target
           self.move_group.set_pose_target(goal.target_pose, self.ee_link)
          
           # Handle collision allowances for pre-grasp approaches
           if goal.allowed_collision_object:
               rospy.loginfo(f"Allowing collision with '{goal.allowed_collision_object}' during planning")
               self.move_group.set_support_surface_name(goal.allowed_collision_object)
           else:
               self.move_group.set_support_surface_name("")  # Clear
          
           rospy.loginfo(f"Planning to target: [{goal.target_pose.position.x:.3f}, "
                        f"{goal.target_pose.position.y:.3f}, {goal.target_pose.position.z:.3f}]")
          
           if not self.publish_feedback(0.4, "Computing motion plan"):
               self.move_group.clear_pose_targets()
               return None
          
           # Plan with timeout monitoring
           plan = self.move_group.plan()
          
           if isinstance(plan, tuple):
               success, trajectory, planning_time, error_code = plan
               rospy.loginfo(f"Planning completed in {planning_time:.2f}s")
           else:
               trajectory = plan
               success = len(trajectory.joint_trajectory.points) > 0
          
           # Clear targets
           self.move_group.clear_pose_targets()
           self.move_group.set_support_surface_name("")
      
       if success:
           num_points = len(trajectory.joint_trajectory.points)
           duration = trajectory.joint_trajectory.points[-1].time_from_start.to_sec() if num_points > 0 else 0.0
          
           if not self.publish_feedback(1.0, f"Planning succeeded ({num_points} waypoints, {duration:.1f}s)"):
               return None
          
           result.success = True
           result.message = f"Planning succeeded with {num_points} waypoints"
           result.trajectory = trajectory
           rospy.loginfo(f"âœ“ Planning complete: {num_points} waypoints, {duration:.1f}s duration")
       else:
           result.success = False
           result.message = "Planning failed - no valid trajectory found"
           rospy.logwarn("âœ— Planning failed")
      
       return result
  
  
   def execute_cb(self, goal):
       """Main action callback with routing to specific handlers"""
       rospy.loginfo(f"â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
       rospy.loginfo(f"Received action: {goal.action}")
      
       try:
           # Route to appropriate handler
           if goal.action == "ATTACH":
               result = self.handle_attach(goal)
           elif goal.action == "DETACH":
               result = self.handle_detach(goal)
           elif goal.action == "HOME":
               result = self.handle_home(goal)
           else:
               # Default: motion planning
               result = self.handle_motion_planning(goal)
          
           # Handle preemption during execution
           if result is None:
               rospy.logwarn("Action preempted during execution")
               return
          
           # Set final result
           if result.success:
               self.server.set_succeeded(result)
           else:
               self.server.set_aborted(result)
              
       except Exception as e:
           result = PlanningActionResult()
           result.success = False
           result.message = f"Exception during {goal.action}: {str(e)}"
           rospy.logerr(result.message)
           self.server.set_aborted(result)
      
       rospy.loginfo(f"â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
  
  
   def shutdown(self):
       """Clean shutdown"""
       rospy.loginfo("Shutting down Planning Action Server")
       moveit_commander.roscpp_shutdown()




if __name__ == '__main__':
   try:
       server = PlanningActionServer()
       rospy.spin()
   except rospy.ROSInterruptException:
       pass
   except Exception as e:
       rospy.logfatal(f"Failed to start Planning Action Server: {e}")
   finally:
       moveit_commander.roscpp_shutdown()






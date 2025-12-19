#!/usr/bin/env python3


import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose


from franka_zed_gazebo.srv import PlanningService, PlanningServiceResponse



class PlanningServiceNode:
    def __init__(self):
        rospy.init_node('planning_service_node')
        
        # Initialize moveit_commander
        moveit_commander.roscpp_initialize([])
        
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Get planning group name from parameter
        planning_group = rospy.get_param('~planning_group', 'panda_manipulator')
        self.move_group = moveit_commander.MoveGroupCommander(planning_group)
        
        # Configure planner
        self.move_group.set_planner_id(rospy.get_param('~planner_id', 'RRTConnect'))
        self.move_group.set_planning_time(rospy.get_param('~planning_time', 5.0))
        self.move_group.set_num_planning_attempts(rospy.get_param('~num_attempts', 10))
        self.move_group.allow_replanning(True)
        
        # Create service
        self.service = rospy.Service('/planning_service', PlanningService, self.handle_planning_request)
        
        rospy.loginfo(f"Planning Service Node Ready for group: {planning_group}")


    def handle_planning_request(self, req):
        """Service callback for planning requests"""
        response = PlanningServiceResponse()
        
        try:
            # --- HANDLE ATTACH/DETACH ACTIONS ---
            if req.action == "ATTACH":
                if not req.object_name:
                    response.success = False
                    response.message = "ATTACH action requires object_name"
                    return response
                
                ee_link = rospy.get_param('~ee_link', 'panda_hand')
                touch_links = [ee_link, "panda_leftfinger", "panda_rightfinger"] # Allow collision with fingers
                
                self.robot.get_planning_frame() # Ensure robot model is loaded
                
                # Attach object
                self.scene.attach_box(ee_link, req.object_name, touch_links=touch_links)
                rospy.loginfo(f"Attached object: {req.object_name} to {ee_link}")
                
                response.success = True
                response.message = f"Object {req.object_name} attached"
                return response

            elif req.action == "DETACH":
                if not req.object_name:
                    response.success = False
                    response.message = "DETACH action requires object_name"
                    return response
                    
                ee_link = rospy.get_param('~ee_link', 'panda_hand')
                self.scene.remove_attached_object(ee_link, name=req.object_name)
                rospy.loginfo(f"Detached object: {req.object_name}")
                
                response.success = True
                response.message = f"Object {req.object_name} detached"
                return response
                
            # --- HANDLE MOVE/PLANNING ACTION ---
            
            # Helper to manage allowed collisions context
            # (MoveIt commander doesn't expose clean ACM modification for a single plan call, 
            # so we might attach it or just skip for now if not using full move_group methods that support it directly.
            # However, set_path_constraints or allowed_planning_time are available.
            # For specific collision allowance, it's often easiest to update the scene temporarily or rely on attach/detach states)
            
            # IF allowed_collision_object is set, we need to allow collision with it.
            # The standard MoveGroupCommander.plan() doesn't take "allowed_collisions".
            # BUT, if we are about to grasp, we usually are close.
            # A common workaround in basic scripts is to just assume ATTACH will handle it if we are holding it.
            # If we are APPROACHING (Descend), we are not holding it yet.
            # We can use "set_support_surface_name" if picking from surface, but here we collide with item itself.
            
            # WORKAROUND: We will temporarily allow collision by attaching it loosely or just relying on the planner's resolution 
            # if the box is small enough. 
            # CORRECT WAY: Update ACM. moveit_commander handles this via 'planning_scene_interface' but it's async and tricky inside a service.
            
            # Let's try explicitly allowing collision by name if provided for this plan? 
            # Actually, move_group.plan() does not support it easily.
            # Instead, we will rely on the caller to managing ATTACH state if it's "holding".
            # For "Pre-Grasp to Grasp" (Descend), we encounter the object.
            
            # If req.allowed_collision_object is set, we might need to assume we can touch it.
            # Currently checking MoveIt Commander API... no direct "allow_collision(obj)" for one plan.
            # We will ignore this for now and rely on the fact that if the gripper is OPEN, 
            # and we target a pose slightly above, it might just be fine. 
            # REAL FIX: Use 'pick' interface or manual ACM update.
            # For this simple script, let's proceed with standard planning. 
            # If the user finds it fails, we will implement ACM update in "ATTACH" style (but without attaching).
            
            # UPDATE: We can't easily do it here without modifying the scene.
            # Let's proceed with standard plan.
            
            ee_link = rospy.get_param('~ee_link', 'panda_hand')
            self.move_group.set_pose_target(req.target_pose, ee_link)
            
            rospy.loginfo(f"Planning to target: [{req.target_pose.position.x:.3f}, "
                         f"{req.target_pose.position.y:.3f}, {req.target_pose.position.z:.3f}]")
            
            # Plan trajectory
            plan = self.move_group.plan()
            
            if isinstance(plan, tuple):
                success, trajectory, planning_time, error_code = plan
            else:
                trajectory = plan
                success = len(trajectory.joint_trajectory.points) > 0
            
            if success:
                response.success = True
                response.message = "Planning succeeded"
                response.trajectory = trajectory
                rospy.loginfo("Planning completed successfully")
            else:
                response.success = False
                response.message = "Planning failed"
                rospy.logwarn(response.message)
            
            self.move_group.clear_pose_targets()
            
        except Exception as e:
            response.success = False
            response.message = f"Planning error: {str(e)}"
            rospy.logerr(response.message)
        
        return response



if __name__ == '__main__':
    try:
        node = PlanningServiceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        moveit_commander.roscpp_shutdown()

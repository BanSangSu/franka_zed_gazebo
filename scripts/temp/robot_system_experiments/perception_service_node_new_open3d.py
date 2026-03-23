#!/usr/bin/env python3


import rospy
import smach
import smach_ros
from geometry_msgs.msg import Pose
import copy


from franka_zed_gazebo.srv import PerceptionService, PlanningService, ControlService, GripperService



class PerceptionState(smach.State):
    """State to call perception service"""
    def __init__(self):
        smach.State.__init__(self, 
                           outcomes=['success', 'failed', 'no_objects'],
                           output_keys=['cube_poses', 'num_cubes', 'cube_dimensions'])
        
    def execute(self, userdata):
        rospy.loginfo('Executing PERCEPTION State')
        
        try:
            # Wait for service
            rospy.wait_for_service('/perception_service', timeout=30.0)
            perception_client = rospy.ServiceProxy('/perception_service', PerceptionService)
            
            # Call service
            response = perception_client(trigger=True)
            
            if response.success:
                if response.num_cubes > 0:
                    # Transform poses for top-down grasp (Flip Z by 180 deg around X)
                    # Perception returns Z-up (world aligned), we want Z-down (into table)
                    
                    # Quaternion for 180 deg rotation around X axis: [1, 0, 0, 0] (x,y,z,w)
                    rot_q = [1.0, 0.0, 0.0, 0.0] 
                    
                    fixed_poses = response.cube_poses
                    for pose in fixed_poses.poses:
                        # Current orientation [x, y, z, w]
                        orig_q = [pose.orientation.x, pose.orientation.y, 
                                 pose.orientation.z, pose.orientation.w]
                        
                        # Apply local rotation: q_new = q_orig * q_rot
                        # (This works because we want to rotate around the object's X axis 
                        #  to flip its Z axis while keeping X aligned)
                        new_q = quaternion_multiply(orig_q, rot_q)
                        
                        pose.orientation.x = new_q[0]
                        pose.orientation.y = new_q[1]
                        pose.orientation.z = new_q[2]
                        pose.orientation.w = new_q[3]
                        
                    userdata.cube_poses = fixed_poses
                    userdata.num_cubes = response.num_cubes
                    userdata.cube_dimensions = response.dimensions  # Store dimensions
                    rospy.loginfo(f"Perception succeeded: {response.num_cubes} cubes detected (Orientations flipped for grasp)")
                    return 'success'
                else:
                    rospy.logwarn("No objects detected")
                    return 'no_objects'
            else:
                rospy.logerr(f"Perception failed: {response.message}")
                return 'failed'
                
        except Exception as e:
            rospy.logerr(f"Perception service call failed: {e}")
            return 'failed'

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions [x, y, z, w].
    res = q1 * q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]


class SelectCubeState(smach.State):
    """State to select the next cube to process"""
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['selected', 'finished'],
                             input_keys=['num_cubes', 'current_cube_idx'],
                             output_keys=['current_cube_idx', 'target_pose'])

    def execute(self, userdata):
        rospy.loginfo('Executing SELECT_CUBE State')
        
        # Check if we have processed all cubes
        if userdata.current_cube_idx >= userdata.num_cubes:
            rospy.loginfo("All cubes processed")
            return 'finished'
            
        rospy.loginfo(f"Selecting cube {userdata.current_cube_idx + 1}/{userdata.num_cubes}")
        return 'selected'


class BasePlanningState(smach.State):
    """Base state for planning"""
    def __init__(self, outcomes=['success', 'failed'], input_keys=['cube_poses', 'current_cube_idx'], output_keys=['planned_trajectory']):
        smach.State.__init__(self, outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        
    def get_target_pose(self, userdata):
        """Override this method to return the specific target pose"""
        raise NotImplementedError
    
    def get_allowed_collision_object(self, userdata):
        """Override this to return name of object allowed to be touched"""
        return ""
        
    def execute(self, userdata):
        state_name = self.__class__.__name__
        rospy.loginfo(f'Executing {state_name}')
        
        try:
            # Wait for service
            rospy.wait_for_service('/planning_service', timeout=5.0)
            planning_client = rospy.ServiceProxy('/planning_service', PlanningService)
            
            target_pose = self.get_target_pose(userdata)
            if target_pose is None:
                return 'failed'
            
            allowed_obj = self.get_allowed_collision_object(userdata)
                
            rospy.loginfo(f"Planning to target: z={target_pose.position.z:.3f} (Allowed Collision: {allowed_obj})")
            
            # Call planning service
            response = planning_client(target_pose=target_pose, 
                                     planning_group='panda_manipulator',
                                     allowed_collision_object=allowed_obj)
            
            if response.success:
                userdata.planned_trajectory = response.trajectory
                rospy.loginfo(f"{state_name} succeeded")
                return 'success'
            else:
                rospy.logerr(f"{state_name} failed: {response.message}")
                return 'failed'
                
        except Exception as e:
            rospy.logerr(f"{state_name} service call failed: {e}")
            return 'failed'


class PlanApproachState(BasePlanningState):
    """Plan trajectory to approach pose (above the object)"""
    def get_target_pose(self, userdata):
        cube_pose = userdata.cube_poses.poses[userdata.current_cube_idx]
        approach_pose = copy.deepcopy(cube_pose)
        approach_pose.position.z += 0.25  # 25cm above
        # Ensure orientation is suitable for top-down grasp (gripper pointing down)
        # Assuming perception returns valid orientation or we overwrite it here
        # For now, trust perception or keep as is
        return approach_pose


class PlanDescendState(BasePlanningState):
    """Plan trajectory to grasp pose (at the object)"""
    def get_target_pose(self, userdata):
        cube_pose = userdata.cube_poses.poses[userdata.current_cube_idx]
        grasp_pose = copy.deepcopy(cube_pose)
        # 0.11m (11cm) from wrist to object center. 
        # Approx: Finger length (~10.3cm) + small gap/overlap.
        grasp_pose.position.z += 0.11 
        return grasp_pose
    
    def get_allowed_collision_object(self, userdata):
        """Allow collision with the target cube while descending"""
        return f"cube_{userdata.current_cube_idx}"


class PlanLiftState(BasePlanningState):
    """Plan trajectory to lift pose (back up)"""
    def get_target_pose(self, userdata):
        cube_pose = userdata.cube_poses.poses[userdata.current_cube_idx]
        lift_pose = copy.deepcopy(cube_pose)
        lift_pose.position.z += 0.25
        return lift_pose
    
    def get_allowed_collision_object(self, userdata):
        return f"cube_{userdata.current_cube_idx}"


class PlanPlaceState(BasePlanningState):
    """Plan trajectory to place pose"""
    def get_target_pose(self, userdata):
        # Place to the right of the object or a predefined bucket
        current_pose = userdata.cube_poses.poses[userdata.current_cube_idx]
        place_pose = copy.deepcopy(current_pose)
        # Logic: Move 20cm in Y relative to world, keep height
        place_pose.position.y += 0.20 
        place_pose.position.z += 0.15 # Drop height
        return place_pose
    
    def get_allowed_collision_object(self, userdata):
        return f"cube_{userdata.current_cube_idx}" 


class SceneState(smach.State):
    """State to interact with the planning scene (Attach/Detach)"""
    def __init__(self, action="ATTACH"):
        smach.State.__init__(self, 
                           outcomes=['success', 'failed'],
                           input_keys=['current_cube_idx'])
        self.action = action
        
    def execute(self, userdata):
        rospy.loginfo(f'Executing SCENE State: {self.action}')
        
        try:
            rospy.wait_for_service('/planning_service', timeout=5.0)
            planning_client = rospy.ServiceProxy('/planning_service', PlanningService)
            
            object_name = f"cube_{userdata.current_cube_idx}"
            
            response = planning_client(action=self.action, object_name=object_name)
            
            if response.success:
                rospy.loginfo(f"Scene action {self.action} succeeded for {object_name}")
                return 'success'
            else:
                rospy.logerr(f"Scene action {self.action} failed: {response.message}")
                return 'failed'
        except Exception as e:
            rospy.logerr(f"Planning service call failed: {e}")
            return 'failed'


class ControlState(smach.State):
    """State to call control service to execute trajectory"""
    def __init__(self):
        smach.State.__init__(self,
                           outcomes=['success', 'failed'],
                           input_keys=['planned_trajectory'])
        
    def execute(self, userdata):
        state_name = self.__class__.__name__
        rospy.loginfo(f'Executing CONTROL State')
        
        try:
            rospy.wait_for_service('/control_service', timeout=30.0)
            control_client = rospy.ServiceProxy('/control_service', ControlService)
            
            response = control_client(trajectory=userdata.planned_trajectory)
            
            if response.success:
                rospy.loginfo("Control execution succeeded")
                return 'success'
            else:
                rospy.logerr(f"Control failed: {response.message}")
                return 'failed'
                
        except Exception as e:
            rospy.logerr(f"Control service call failed: {e}")
            return 'failed'


class GripperState(smach.State):
    """State to control the gripper with optional force and width control"""
    def __init__(self, open_gripper=True, grasp_force=20.0):
        # Input keys only needed when closing (to get cube dimensions)
        if open_gripper:
            smach.State.__init__(self, outcomes=['success', 'failed'])
        else:
            smach.State.__init__(self, outcomes=['success', 'failed'],
                                input_keys=['cube_dimensions', 'current_cube_idx'])
        self.open_gripper = open_gripper
        self.grasp_force = grasp_force
        
    def execute(self, userdata):
        action = "OPEN" if self.open_gripper else "CLOSE"
        rospy.loginfo(f'Executing GRIPPER State: {action}')
        
        try:
            rospy.wait_for_service('/gripper_service', timeout=5.0)
            gripper_client = rospy.ServiceProxy('/gripper_service', GripperService)
            
            if self.open_gripper:
                # Open: just send open=True
                response = gripper_client(open=True, width=0.0, force=0.0)
            else:
                # Close/Grasp: Calculate target width from cube dimensions
                # Get the smallest dimension of the cube as the grasp width
                try:
                    dims = userdata.cube_dimensions[userdata.current_cube_idx]
                    # Use the smaller of x,y as grasp width (we grasp from sides)
                    target_width = min(dims.x, dims.y)
                    # Add small margin to not crush the object
                    target_width = max(0.01, target_width - 0.005)  # 5mm less than object
                    rospy.loginfo(f"Grasping cube with dimensions ({dims.x:.3f}, {dims.y:.3f}, {dims.z:.3f}), target_width={target_width:.3f}m")
                except Exception as e:
                    rospy.logwarn(f"Could not get cube dimensions: {e}. Using default width.")
                    target_width = 0.03  # Default 3cm
                
                response = gripper_client(open=False, width=target_width, force=self.grasp_force)
            
            if response.success:
                rospy.loginfo(f"Gripper {action} succeeded (final_width={response.final_width:.4f}m)")
                return 'success'
            else:
                rospy.logerr(f"Gripper {action} failed: {response.message}")
                return 'failed'
        except Exception as e:
            rospy.logerr(f"Gripper service call failed: {e}")
            return 'failed'


class UpdateIndexState(smach.State):
    """State to increment cube index"""
    def __init__(self):
        smach.State.__init__(self, outcomes=['success'], 
                             input_keys=['current_cube_idx'],
                             output_keys=['current_cube_idx'])
        
    def execute(self, userdata):
        userdata.current_cube_idx += 1
        return 'success'


def main():
    rospy.init_node('robot_orchestrator')
    
    # Create top-level SMACH state machine
    sm = smach.StateMachine(outcomes=['succeeded', 'failed', 'aborted'])
    
    # Initialize user data
    sm.userdata.current_cube_idx = 0
    sm.userdata.num_cubes = 0
    
    # Build state machine
    with sm:
        # PERCEPTION: Detect cubes
        smach.StateMachine.add('PERCEPTION',
                              PerceptionState(),
                              transitions={'success': 'SELECT_CUBE',
                                         'failed': 'aborted',
                                         'no_objects': 'PERCEPTION'},
                              remapping={'cube_poses': 'cube_poses',
                                       'num_cubes': 'num_cubes'})
        
        # SELECT_CUBE: Choose next cube
        smach.StateMachine.add('SELECT_CUBE',
                              SelectCubeState(),
                              transitions={'selected': 'PLAN_APPROACH',
                                         'finished': 'succeeded'},
                              remapping={'num_cubes': 'num_cubes',
                                       'current_cube_idx': 'current_cube_idx'})
        
        # --- PICK SEQUENCE ---
        
        # 1. PLAN APPROACH
        smach.StateMachine.add('PLAN_APPROACH',
                              PlanApproachState(),
                              transitions={'success': 'EXECUTE_APPROACH',
                                         'failed': 'PERCEPTION'},
                              remapping={'cube_poses': 'cube_poses',
                                         'current_cube_idx': 'current_cube_idx',
                                         'planned_trajectory': 'planned_trajectory'})
        
        # 2. EXECUTE APPROACH
        smach.StateMachine.add('EXECUTE_APPROACH',
                              ControlState(),
                              transitions={'success': 'OPEN_GRIPPER',
                                         'failed': 'PERCEPTION'},
                              remapping={'planned_trajectory': 'planned_trajectory'})
                              
        # 3. OPEN GRIPPER
        smach.StateMachine.add('OPEN_GRIPPER',
                              GripperState(open_gripper=True),
                              transitions={'success': 'PLAN_DESCEND',
                                           'failed': 'PERCEPTION'})

        # 4. PLAN DESCEND
        smach.StateMachine.add('PLAN_DESCEND',
                              PlanDescendState(),
                              transitions={'success': 'EXECUTE_DESCEND',
                                         'failed': 'PERCEPTION'},
                              remapping={'cube_poses': 'cube_poses',
                                         'current_cube_idx': 'current_cube_idx',
                                         'planned_trajectory': 'planned_trajectory'})

        # 5. EXECUTE DESCEND
        smach.StateMachine.add('EXECUTE_DESCEND',
                              ControlState(),
                              transitions={'success': 'CLOSE_GRIPPER',
                                         'failed': 'PERCEPTION'},
                              remapping={'planned_trajectory': 'planned_trajectory'})

        # 6. CLOSE GRIPPER (Grasp) with force control
        # If failed (missed), we skip to OPEN and Next
        smach.StateMachine.add('CLOSE_GRIPPER',
                              GripperState(open_gripper=False, grasp_force=20.0), # Increased from 15.0 to 50.0
                              transitions={'success': 'ATTACH_OBJECT',
                                           'failed': 'OPEN_AND_SKIP'},
                              remapping={'cube_dimensions': 'cube_dimensions',
                                         'current_cube_idx': 'current_cube_idx'})
        
        # 6-B. ATTACH OBJECT (Only if grasped)
        smach.StateMachine.add('ATTACH_OBJECT',
                               SceneState(action="ATTACH"),
                               transitions={'success': 'PLAN_LIFT',
                                            'failed': 'PERCEPTION'},
                               remapping={'current_cube_idx': 'current_cube_idx'})
                                           
        # 7. PLAN LIFT
        smach.StateMachine.add('PLAN_LIFT',
                              PlanLiftState(),
                              transitions={'success': 'EXECUTE_LIFT',
                                         'failed': 'PERCEPTION'},
                              remapping={'cube_poses': 'cube_poses',
                                         'current_cube_idx': 'current_cube_idx',
                                         'planned_trajectory': 'planned_trajectory'})

        # 8. EXECUTE LIFT
        smach.StateMachine.add('EXECUTE_LIFT',
                              ControlState(),
                              transitions={'success': 'PLAN_PLACE',
                                         'failed': 'PERCEPTION'},
                              remapping={'planned_trajectory': 'planned_trajectory'})

        # --- PLACE SEQUENCE ---
        
        # 9. PLAN PLACE
        smach.StateMachine.add('PLAN_PLACE',
                               PlanPlaceState(),
                               transitions={'success': 'EXECUTE_PLACE',
                                            'failed': 'PERCEPTION'},
                               remapping={'cube_poses': 'cube_poses',
                                          'current_cube_idx': 'current_cube_idx',
                                          'planned_trajectory': 'planned_trajectory'})
        
        # 10. EXECUTE PLACE
        smach.StateMachine.add('EXECUTE_PLACE',
                               ControlState(),
                               transitions={'success': 'OPEN_GRIPPER_PLACE',
                                            'failed': 'PERCEPTION'},
                               remapping={'planned_trajectory': 'planned_trajectory'})
        
        # 11. OPEN GRIPPER (Release)
        smach.StateMachine.add('OPEN_GRIPPER_PLACE',
                               GripperState(open_gripper=True),
                               transitions={'success': 'DETACH_OBJECT',
                                            'failed': 'PERCEPTION'})
                                            
        # 12. DETACH OBJECT
        smach.StateMachine.add('DETACH_OBJECT',
                               SceneState(action="DETACH"),
                               transitions={'success': 'UPDATE_INDEX',
                                            'failed': 'UPDATE_INDEX'}, # Even if detach fails, try next
                               remapping={'current_cube_idx': 'current_cube_idx'})

        # Error Recovery: OPEN AND SKIP
        smach.StateMachine.add('OPEN_AND_SKIP',
                               GripperState(open_gripper=True),
                               transitions={'success': 'UPDATE_INDEX',
                                            'failed': 'UPDATE_INDEX'})

        # 13. UPDATE INDEX
        smach.StateMachine.add('UPDATE_INDEX',
                              UpdateIndexState(),
                              transitions={'success': 'SELECT_CUBE'},
                              remapping={'current_cube_idx': 'current_cube_idx'})

    
    # Create and start introspection server for visualization
    sis = smach_ros.IntrospectionServer('robot_state_machine', sm, '/SM_ROOT')
    sis.start()
    
    # Execute state machine
    rospy.loginfo("Starting Robot Orchestrator State Machine")
    outcome = sm.execute()
    rospy.loginfo(f"State machine finished with outcome: {outcome}")
    
    # Stop introspection server
    sis.stop()
    rospy.spin()



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

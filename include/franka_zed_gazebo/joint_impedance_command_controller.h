#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// ROS
#include <franka/robot_state.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

// controller_interface MUST come before franka_hw includes
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>

// franka_hw AFTER controller_interface
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

// realtime tools
#include <realtime_tools/realtime_buffer.h>

namespace franka_zed_gazebo {

class JointImpedanceCommandController
    : public controller_interface::MultiInterfaceController<
          franka_hw::FrankaModelInterface,
          hardware_interface::EffortJointInterface,
          franka_hw::FrankaStateInterface>
{
public:
    bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& nh) override;
    void starting(const ros::Time& time) override;
    void update(const ros::Time& time, const ros::Duration& period) override;

private:
    std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
    std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;

    std::vector<std::string> joint_names_;  
    std::vector<hardware_interface::JointHandle> joint_handles_;

    std::array<double, 7> k_gains_{};
    std::array<double, 7> d_gains_{};
    
    std::array<double, 7> tau_prev_{};
    std::array<double, 7> tau_filtered_{};
    std::array<double, 7> dq_filtered_{}; // ADDED: Velocity filter history
    bool first_update_{true};

    struct DesiredState {
        std::array<double, 7> q_d{};
        std::array<double, 7> dq_d{};
        bool valid{false};
    };
    realtime_tools::RealtimeBuffer<DesiredState> desired_state_buffer_;

    std::array<double, 7> q_start_{};

    ros::Subscriber command_sub_;

    void commandCallback(const sensor_msgs::JointState::ConstPtr& msg);
};

} // namespace franka_zed_gazebo
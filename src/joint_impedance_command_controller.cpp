#include <franka_zed_gazebo/joint_impedance_command_controller.h>
#include <pluginlib/class_list_macros.h>
#include <franka/robot_state.h>
#include <algorithm> // For std::clamp

namespace franka_zed_gazebo {

bool JointImpedanceCommandController::init(hardware_interface::RobotHW* robot_hw,
                                            ros::NodeHandle& nh)
{
    // --- Franka Model Interface ---
    auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
    if (!model_interface) {
        ROS_ERROR("FrankaModelInterface not found");
        return false;
    }
    try {
        model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
            model_interface->getHandle("panda_model"));
    } catch (const hardware_interface::HardwareInterfaceException& e) {
        ROS_ERROR_STREAM("Cannot get model handle: " << e.what());
        return false;
    }

    // --- Franka State Interface ---
    auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
    if (!state_interface) {
        ROS_ERROR("FrankaStateInterface not found");
        return false;
    }
    try {
        state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
            state_interface->getHandle("panda_robot"));
    } catch (const hardware_interface::HardwareInterfaceException& e) {
        ROS_ERROR_STREAM("Cannot get state handle: " << e.what());
        return false;
    }

    // --- Effort Joint Interface (7 joints) ---
    auto* effort_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
    if (!effort_interface) {
        ROS_ERROR("EffortJointInterface not found");
        return false;
    }

    if (!nh.getParam("joint_names", joint_names_) || joint_names_.size() != 7) {
        ROS_ERROR("Need exactly 7 joint_names");
        return false;
    }

    for (const auto& name : joint_names_) {
        try {
            joint_handles_.push_back(effort_interface->getHandle(name));
        } catch (const hardware_interface::HardwareInterfaceException& e) {
            ROS_ERROR_STREAM("Cannot get joint handle " << name << ": " << e.what());
            return false;
        }
    }

    // --- Gains from parameter server ---
    std::vector<double> k, d;
    nh.param<std::vector<double>>("k_gains", k, {600, 600, 600, 600, 250, 150, 50});
    nh.param<std::vector<double>>("d_gains", d, { 50,  50,  50,  50,  30,  25, 15});
    for (int i = 0; i < 7; i++) {
        k_gains_[i] = k[i];
        d_gains_[i] = d[i];
    }

    // --- Subscribe to your Python node's topic ---
    command_sub_ = nh.subscribe(
        "/joint_impedance_controller/joint_command", 1,
        &JointImpedanceCommandController::commandCallback, this);

    ROS_INFO("JointImpedanceCommandController initialized");
    return true;
}

void JointImpedanceCommandController::starting(const ros::Time&)
{
    // Capture current position as safe hold target
    franka::RobotState state = state_handle_->getRobotState();
    
    for (int i = 0; i < 7; i++) {
        q_start_[i] = state.q[i];
        dq_filtered_[i] = state.dq[i]; // FIX: Initialize velocity filter to current velocity
        
        // FIX: Franka handles gravity internally. Start commanded torques at 0.0 to prevent jerking.
        tau_prev_[i]     = 0.0;
        tau_filtered_[i] = 0.0;
    }
    first_update_ = true;

    DesiredState init;
    init.q_d  = q_start_;
    init.dq_d = {};
    init.valid = true;
    desired_state_buffer_.writeFromNonRT(init);

    ROS_INFO_STREAM("JointImpedanceCommandController::starting() — holding q: "
        << q_start_[0] << " " << q_start_[1] << " " << q_start_[2] << " "
        << q_start_[3] << " " << q_start_[4] << " " << q_start_[5] << " "
        << q_start_[6]);
}

void JointImpedanceCommandController::update(const ros::Time&, const ros::Duration&)
{
    franka::RobotState state = state_handle_->getRobotState();
    const DesiredState& des  = *desired_state_buffer_.readFromRT();

    std::array<double,7> coriolis = model_handle_->getCoriolis();

    // Filter constants
    const double alpha_dq      = 0.99; // Velocity filter constant
    const double delta_tau_max = 1.0;  // Max torque change per 1ms tick

    for (int i = 0; i < 7; i++) {
        // 1. Filter the measured velocity to prevent D-gain vibrations
        dq_filtered_[i] = (1.0 - alpha_dq) * dq_filtered_[i] + alpha_dq * state.dq[i];

        double q_err  = des.q_d[i]  - state.q[i];
        double dq_err = des.dq_d[i] - dq_filtered_[i]; // Use filtered velocity

        // 2. Compute PD control law (FIX: Removed gravity[i] from this equation!)
        double tau_desired = k_gains_[i] * q_err
                           + d_gains_[i] * dq_err
                           + coriolis[i];

        // 3. Rate-limit the torque difference based on the robot's actual last desired torque
        double diff = std::clamp(tau_desired - state.tau_J_d[i], -delta_tau_max, delta_tau_max);
        double tau_out = state.tau_J_d[i] + diff;

        // 4. Update local history
        tau_prev_[i] = tau_out;

        // 5. Clamp to absolute hardware safety limits and send command
        tau_out = std::clamp(tau_out, -60.0, 60.0);
        joint_handles_[i].setCommand(tau_out);
    }
}

void JointImpedanceCommandController::commandCallback(
    const sensor_msgs::JointState::ConstPtr& msg)
{
    if (msg->position.size() < 7) {
        ROS_WARN_THROTTLE(1.0, "JointCommand: expected 7 positions, got %zu",
                          msg->position.size());
        return;
    }

    DesiredState des;

    if (msg->name.size() >= 7) {
        // Build a name→index map for our handles
        std::unordered_map<std::string, int> name_to_idx;
        for (int i = 0; i < 7; i++) {
            name_to_idx[joint_names_[i]] = i;
        }

        bool all_found = true;
        for (int j = 0; j < 7; j++) {
            auto it = name_to_idx.find(msg->name[j]);
            if (it == name_to_idx.end()) {
                ROS_WARN_THROTTLE(1.0, "Unknown joint name in command: %s",
                                  msg->name[j].c_str());
                all_found = false;
                break;
            }
            int idx       = it->second;
            des.q_d[idx]  = msg->position[j];
            des.dq_d[idx] = (msg->velocity.size() >= 7) ? msg->velocity[j] : 0.0;
        }

        if (!all_found) return;   // drop malformed message

    } else {
        // Fallback: assume positional ordering matches joint_names_
        ROS_WARN_ONCE("JointCommand has no joint names — assuming positional order matches.");
        for (int i = 0; i < 7; i++) {
            des.q_d[i]  = msg->position[i];
            des.dq_d[i] = (msg->velocity.size() >= 7) ? msg->velocity[i] : 0.0;
        }
    }

    des.valid = true;
    desired_state_buffer_.writeFromNonRT(des);
}

} // namespace franka_zed_gazebo

PLUGINLIB_EXPORT_CLASS(franka_zed_gazebo::JointImpedanceCommandController,
                       controller_interface::ControllerBase)
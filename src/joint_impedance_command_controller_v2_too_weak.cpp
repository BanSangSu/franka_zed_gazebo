#include <franka_zed_gazebo/joint_impedance_command_controller.h>
#include <pluginlib/class_list_macros.h>
#include <algorithm>

namespace franka_zed_gazebo {

bool JointImpedanceCommandController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& nh) {
    auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
    auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
    auto* effort_interface = robot_hw->get<hardware_interface::EffortJointInterface>();

    if (!model_interface || !state_interface || !effort_interface) return false;

    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(model_interface->getHandle("panda_model"));
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(state_interface->getHandle("panda_robot"));

    if (!nh.getParam("joint_names", joint_names_) || joint_names_.size() != 7) return false;

    for (const auto& name : joint_names_) {
        joint_handles_.push_back(effort_interface->getHandle(name));
    }

    std::vector<double> k, d;
    nh.param<std::vector<double>>("k_gains", k, {600, 600, 600, 600, 250, 150, 50});
    nh.param<std::vector<double>>("d_gains", d, {50, 50, 50, 50, 30, 25, 15});
    for (int i = 0; i < 7; i++) {
        k_gains_[i] = k[i];
        d_gains_[i] = d[i];
    }

    command_sub_ = nh.subscribe("/joint_impedance_controller/joint_command", 1, &JointImpedanceCommandController::commandCallback, this);
    return true;
}

void JointImpedanceCommandController::starting(const ros::Time&) {
    franka::RobotState state = state_handle_->getRobotState();
    for (int i = 0; i < 7; i++) {
        q_d_filtered_[i] = state.q[i]; 
        dq_filtered_[i] = 0.0;
    }
    DesiredState init;
    init.q_d = q_d_filtered_;
    init.valid = true;
    desired_state_buffer_.writeFromNonRT(init);
}

void JointImpedanceCommandController::update(const ros::Time&, const ros::Duration&) {
    franka::RobotState state = state_handle_->getRobotState();
    const DesiredState& des = *desired_state_buffer_.readFromRT();

    std::array<double, 7> coriolis = model_handle_->getCoriolis();
    std::array<double, 7> gravity = model_handle_->getGravity();

    double alpha_v = 0.99; // Velocity filter constant
    double alpha_q = 0.95; // Command smoothing constant

    std::array<double, 7> tau_d_calculated;
    // for (int i = 0; i < 7; i++) {
    //     // Filter measured velocity to reduce noise
    //     dq_filtered_[i] = (1 - alpha_v) * dq_filtered_[i] + alpha_v * state.dq[i];
        
    //     // Smooth incoming command to prevent jumps from Python network jitter
    //     q_d_filtered_[i] = alpha_q * q_d_filtered_[i] + (1 - alpha_q) * des.q_d[i];

    //     double q_err = q_d_filtered_[i] - state.q[i];
    //     double dq_err = des.dq_d[i] - dq_filtered_[i];

    //     tau_d_calculated[i] = k_gains_[i] * q_err + d_gains_[i] * dq_err + coriolis[i];
    // }
    ///
    double max_execution_error = 0.2; // Radians. If pushed further than this, stop fighting.

    for (int i = 0; i < 7; i++) {
        // 1. Update filters (Crucial: otherwise des.q_d is ignored!)
        dq_filtered_[i] = (1 - alpha_v) * dq_filtered_[i] + alpha_v * state.dq[i];
        q_d_filtered_[i] = alpha_q * q_d_filtered_[i] + (1 - alpha_q) * des.q_d[i];

        // 2. Compliance Logic (The "Push" limit)
        double q_err = q_d_filtered_[i] - state.q[i];
        if (std::abs(q_err) > max_execution_error) {
            // If pushed too far, shift the "target" so the robot doesn't fight back violently
            q_d_filtered_[i] = state.q[i] + (std::signbit(q_err) ? -max_execution_error : max_execution_error);
            q_err = q_d_filtered_[i] - state.q[i]; // Recalculate clipped error
        }

        // 3. Torque calculation
        double dq_err = des.dq_d[i] - dq_filtered_[i];
        tau_d_calculated[i] = k_gains_[i] * q_err + d_gains_[i] * dq_err + coriolis[i];
    }
    ///

    // Rate-limit torque changes relative to the robot's last commanded state
    std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, state.tau_J_d);

    for (int i = 0; i < 7; ++i) {
        joint_handles_[i].setCommand(tau_d_saturated[i]);
    }
}

std::array<double, 7> JointImpedanceCommandController::saturateTorqueRate(
    const std::array<double, 7>& tau_d_calculated,
    const std::array<double, 7>& tau_J_d) {
    std::array<double, 7> tau_d_saturated{};
    for (size_t i = 0; i < 7; i++) {
        double diff = tau_d_calculated[i] - tau_J_d[i];
        tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(diff, kDeltaTauMax), -kDeltaTauMax);
    }
    return tau_d_saturated;
}

void JointImpedanceCommandController::commandCallback(const sensor_msgs::JointState::ConstPtr& msg) {
    if (msg->position.size() < 7) return;
    DesiredState des;
    // Map names to indices to ensure correct joint ordering
    std::unordered_map<std::string, int> name_map;
    for(int i=0; i<7; ++i) name_map[joint_names_[i]] = i;

    for (size_t j = 0; j < msg->name.size(); j++) {
        if (name_map.count(msg->name[j])) {
            int idx = name_map[msg->name[j]];
            des.q_d[idx] = msg->position[j];
            des.dq_d[idx] = (msg->velocity.size() > j) ? msg->velocity[j] : 0.0;
        }
    }
    des.valid = true;
    desired_state_buffer_.writeFromNonRT(des);
}

} // namespace franka_zed_gazebo

PLUGINLIB_EXPORT_CLASS(franka_zed_gazebo::JointImpedanceCommandController, controller_interface::ControllerBase)
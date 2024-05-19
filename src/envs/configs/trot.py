"""Config for Go1 speed tracking environment."""

from ml_collections import ConfigDict
import numpy as np
import torch


def get_config():
    config = ConfigDict()

    gait_config = ConfigDict()
    gait_config.stepping_frequency = 1.5
    gait_config.initial_offset = np.array(
        [0.05, 0.55, 0.55, 0.05], dtype=np.float32
    ) * (2 * np.pi)
    gait_config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    config.gait = gait_config

    config.goal_lb = torch.tensor(
        [-1.0, -0.5, -1.0], dtype=torch.float
    )  # Lin_x, Lin_y, Rot_z
    config.goal_ub = torch.tensor([1.0, 0.5, 1.0], dtype=torch.float)

    # Action: step_freq, height, vx, vy,  vz, roll, pitch, pitch_rate, yaw_rate
    config.include_gait_action = True
    config.include_foot_action = True
    config.mirror_foot_action = True

    # Fully Flexible
    config.action_lb = np.array(
        [0.5, -0.001, -3.0, -3.0, -3.0, -0.001, -0.001, -3.0, -2.5, -3.0]
        + [-0.1, -0.1, -0.1] * 2
    )
    config.action_ub = np.array(
        [1.5, 0.001, 3.0, 3.0, 3.0, 0.001, 0.001, 3.0, 2.5, 3.0] + [0.1, 0.1, 0.1] * 2
    )

    config.episode_length_s = 20.0
    config.max_jumps = 10.0
    config.env_dt = 0.01
    config.motor_strength_ratios = 1.0
    config.motor_torque_delay_steps = 5
    config.use_yaw_feedback = False
    config.foot_friction = 0.7  # 0.7
    config.base_position_kp = np.array([0.0, 0.0, 0.0])
    config.base_position_kd = np.array([10.0, 10.0, 10.0])
    config.base_orientation_kp = np.array([0.0, 0.0, 0.0])
    config.base_orientation_kd = np.array([10.0, 10.0, 10.0])
    config.qp_foot_friction_coef = 0.5
    config.qp_weight_ddq = np.diag([1.0, 1.0, 10.0, 10.0, 10.0, 1.0])
    config.qp_body_mass = 13.076
    config.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 1.5

    config.clip_grf_in_sim = False
    config.use_full_qp = True
    config.warm_up = True
    config.iter = 20

    config.swing_foot_height = 0.1
    config.swing_foot_landing_clearance = 0.0
    config.terminate_on_body_contact = True
    config.terminate_on_limb_contact = True
    config.terminate_on_height = 0.15
    config.use_penetrating_contact = False

    config.rewards = [
        ("upright", 0.02),
        ("contact_consistency", 0.008),
        ("foot_slipping", 0.032),
        ("foot_clearance", 0.008),
        ("out_of_bound_action", 0.01),
        ("knee_contact", 0.064),
        ("stepping_freq", 0.008),
        ("speed_tracking", 0.016),
        ("height", 0.06),
        ("lin_vel_z", 0.01),
        ("ang_vel_xy", 0.003),
    ]
    config.clip_negative_reward = False
    config.normalize_reward_by_phase = True

    config.terminal_rewards = []
    config.clip_negative_terminal_reward = False
    return config

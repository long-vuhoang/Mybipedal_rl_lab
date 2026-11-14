import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from bipedal_locomotion.actuators import DelayedImplicitActuatorCfg

current_dir = os.path.dirname(__file__)
urdf_path = os.path.join(current_dir, "../usd/SF_MYBIPEDAL/urdf/mybipedal_real_normalized_rev.urdf")

SOLEFOOT_MYBIPEDAL_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{urdf_path}",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            # left leg
            "left_hip_pitch_joint": -0.2,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": -0.5,
            "left_ankle_pitch_joint": -0.3,
            # right leg
            "right_hip_pitch_joint": -0.2,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": -0.5,
            "right_ankle_pitch_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "RS05": DelayedImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            effort_limit_sim=6.0,
            velocity_limit_sim=14.0,
            stiffness=30.0,
            damping=1.5,
            armature=0.00625, # Apply small armature to avoid oscillations in the hip yaw joint RS05
            min_delay=0,
            max_delay=4,
        ),
        "RS02": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
            ],
            effort_limit_sim=17.0,
            velocity_limit_sim=12.57,
            stiffness={
                ".*_hip_roll_joint": 80.0,
                ".*_hip_pitch_joint": 80.0,
            },
            damping={
                ".*_hip_roll_joint": 2.3,
                ".*_hip_pitch_joint": 2.3,
            },
            armature=0.00625,
            min_delay=0,
            max_delay=4,
        ),
        "RS00_knee": DelayedImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit_sim=14.0,
            velocity_limit_sim=12.57,
            stiffness=80.0,
            damping=2.3,
            armature=0.00625,
            min_delay=0,
            max_delay=4,
        ),
        "RS00_ankle": DelayedImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=14.0,
            velocity_limit_sim=14.0,
            stiffness=50.0,
            damping=1.8,
            armature=0.00625,
            min_delay=0,
            max_delay=4,
        ),
    },
)

import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.mybipedal_rsl_rl_ppo_cfg import (
    SF_MYBIPEDALFlatPPORunnerCfg, WF_MYBIPEDALFlatPPORunnerCfg,
)

from . import solefoot_mybipedal_env_cfg, wheelfoot_mybipedal_env_cfg

##
# Create PPO runners for RSL-RL
##

sf_mybipedal_blind_flat_runner_cfg = SF_MYBIPEDALFlatPPORunnerCfg()

wf_mybipedal_blind_flat_runner_cfg = WF_MYBIPEDALFlatPPORunnerCfg()

##
# Register Gym environments
##


######################################
# SF_MYBIPEDAL Blind Flat Environment
######################################
gym.register(
    id="SF-MYBIPEDAL-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_mybipedal_env_cfg.SF_MYBIPEDAL_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": sf_mybipedal_blind_flat_runner_cfg,
    },
)

gym.register(
    id="SF-MYBIPEDAL-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_mybipedal_env_cfg.SF_MYBIPEDAL_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": sf_mybipedal_blind_flat_runner_cfg,
    },
)


######################################
# WF_MYBIPEDAL Blind Flat Environment
######################################
gym.register(
    id="WF-MYBIPEDAL-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_mybipedal_env_cfg.WF_MYBIPEDAL_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": wf_mybipedal_blind_flat_runner_cfg,
    },
)

gym.register(
    id="WF-MYBIPEDAL-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_mybipedal_env_cfg.WF_MYBIPEDAL_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_mybipedal_blind_flat_runner_cfg,
    },
)
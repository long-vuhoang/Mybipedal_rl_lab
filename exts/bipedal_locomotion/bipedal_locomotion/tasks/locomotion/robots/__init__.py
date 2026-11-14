import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.minibot_rsl_rl_ppo_cfg import (
    SF_MINIBOTFlatPPORunnerCfg, WF_MINIBOTFlatPPORunnerCfg,
)

from . import solefoot_minibot_env_cfg, wheelfoot_minibot_env_cfg

##
# Create PPO runners for RSL-RL
##

sf_minibot_blind_flat_runner_cfg = SF_MINIBOTFlatPPORunnerCfg()

wf_minibot_blind_flat_runner_cfg = WF_MINIBOTFlatPPORunnerCfg()

##
# Register Gym environments
##


######################################
# SF_MINIBOT Blind Flat Environment
######################################
gym.register(
    id="SF-MINIBOT-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_minibot_env_cfg.SF_MINIBOT_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": sf_minibot_blind_flat_runner_cfg,
    },
)

gym.register(
    id="SF-MINIBOT-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_minibot_env_cfg.SF_MINIBOT_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": sf_minibot_blind_flat_runner_cfg,
    },
)


######################################
# WF_MINIBOT Blind Flat Environment
######################################
gym.register(
    id="WF-MINIBOT-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_minibot_env_cfg.WF_MINIBOT_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": wf_minibot_blind_flat_runner_cfg,
    },
)

gym.register(
    id="WF-MINIBOT-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_minibot_env_cfg.WF_MINIBOT_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_minibot_blind_flat_runner_cfg,
    },
)
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Script to play a checkpoint of an RL agent from RSL-RL, with optional keyboard
velocity teleop (W/A/S/D/Q/E/X/R) when not running headless.

This is a variant of play.py that additionally wires up `Keyboard` from
`bipedal_locomotion.utils.keyboard` so the `base_velocity` command term can be
driven manually in the viewer instead of being sampled randomly.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl")))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL, with optional keyboard teleop.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument(
    "--no_keyboard",
    action="store_true",
    default=False,
    help="Disable keyboard velocity teleop even when not running headless.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import (
    RslRlPpoAlgorithmMlpCfg,
    export_mlp_as_onnx,
    export_policy_as_jit,
    export_encoder_as_jit,
)


def main():
    """Play with RSL-RL agent, optionally driven by keyboard velocity commands."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # when teleoperating with the keyboard, disable the heading controller on the
    # base_velocity command term -- otherwise command_manager overwrites ang_vel_z
    # every step from the sampled heading target, and Q/E keys have no visible effect
    use_keyboard = (not args_cli.headless) and (not args_cli.no_keyboard)
    if use_keyboard and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.heading_command = False

    # specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_encoder_as_jit(
            ppo_runner.alg.encoder, export_model_dir
        )
        print("Exported encoder as jit script to: ", export_model_dir)
        # keep onnx exports too -- used by the sim2real deployment stack
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            export_model_dir,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    # set up keyboard velocity teleop (viewer only, i.e. not headless)
    if use_keyboard:
        from bipedal_locomotion.utils.keyboard import Keyboard
        keyboard = Keyboard(env, command_name="base_velocity")  # noqa:F841

    # reset environment
    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict.get("commands")
    robot = env.unwrapped.scene["robot"]

    # tìm body id của 2 chân — sửa regex theo tên khớp/body thực tế trong USD của bạn
    feet_ids, feet_names = robot.find_bodies(".*ankle_pitch.*")   # vd: ["left_foot", "right_foot"]
    print(f"[INFO] feet bodies found: {feet_names}")

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            # env stepping
            obs_dict, _, _, infos = env.step(actions)
            base_height = robot.data.root_pos_w[:, 2]                     # (num_envs,)
            feet_pos = robot.data.body_pos_w[:, feet_ids, :]              # (num_envs, 2, 3)
            # khoảng cách 3D
            feet_dist = torch.norm(feet_pos[:, 0, :] - feet_pos[:, 1, :], dim=-1)
            # nếu chỉ muốn khoảng cách ngang (xy), bỏ trục z:
            feet_dist_xy = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
            
            print(f"base_h={base_height[0].item():.3f} m | feet_dist={feet_dist[0].item():.3f} m | feet_dist_xy={feet_dist_xy[0].item():.3f} m")
            obs = obs_dict["policy"]
            obs_history = obs_dict.get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = obs_dict.get("commands")

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
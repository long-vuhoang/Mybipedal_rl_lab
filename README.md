# mybipedal_rl_lab

Reinforcement-learning training stack for the **Mybipedal** bipedal robot
(sole-foot and wheel-foot variants), built on NVIDIA Isaac Lab and RSL-RL.

Derived from LimX Dynamics' open-source `tron2_rl_lab` (Apache-2.0). All
TRON2A robot assets, configs and tasks have been removed; see `NOTICE`,
`LICENSE` and `THIRD_PARTY_NOTICES.md` for attribution.

## Layout

```
exts/bipedal_locomotion/bipedal_locomotion/
├── assets/
│   ├── config/solefoot_mybipedal_cfg.py      # SF Mybipedal articulation / actuators
│   ├── config/wheelfoot_mybipedal_cfg.py     # WF Mybipedal articulation / actuators
│   └── usd/SF_MYBIPEDAL/                     # meshes, URDF, MuJoCo XML
├── tasks/locomotion/
│   ├── agents/mybipedal_rsl_rl_ppo_cfg.py    # PPO runner cfgs
│   ├── cfg/SF_MYBIPEDAL/, cfg/WF_MYBIPEDAL/    # base env cfgs (scene, obs, rewards, ...)
│   ├── robots/                             # env cfgs + gym registration
│   └── mdp/                                # rewards, observations, events, commands
└── utils/                                  # keyboard teleop, rsl_rl wrappers/export
rsl_rl/                                     # vendored RSL-RL fork
scripts/rsl_rl/                             # train.py, play.py, play_control.py
```

## Install

```bash
# inside your Isaac Lab python environment
pip install -e exts/bipedal_locomotion
pip install -e rsl_rl
```

## Tasks

| Task ID                          | Robot             |
|----------------------------------|-------------------|
| `SF-MYBIPEDAL-Blind-Flat-v0`       | Mybipedal sole-foot  |
| `SF-MYBIPEDAL-Blind-Flat-Play-v0`  | Mybipedal sole-foot (play) |
| `WF-MYBIPEDAL-Blind-Flat-v0`       | Mybipedal wheel-foot |
| `WF-MYBIPEDAL-Blind-Flat-Play-v0`  | Mybipedal wheel-foot (play) |

## Train

```bash
python scripts/rsl_rl/train.py --task SF-MYBIPEDAL-Blind-Flat-v0 --num_envs 4096 --headless
```

## Play

```bash
python scripts/rsl_rl/play.py --task SF-MYBIPEDAL-Blind-Flat-Play-v0 --num_envs 32
# with keyboard velocity teleop
python scripts/rsl_rl/play_control.py --task SF-MYBIPEDAL-Blind-Flat-Play-v0 --num_envs 1
```

Training outputs go to `logs/rsl_rl/` and are git-ignored, as are
`*.pt`, `*.onnx` and `*.jit` files.

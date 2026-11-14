from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.solefoot_mybipedal_cfg import SOLEFOOT_MYBIPEDAL_CFG
from bipedal_locomotion.tasks.locomotion.cfg.SF_MYBIPEDAL.mybipedal_base_env_cfg import SF_MYBIPEDAL_EnvCfg


######################
# SF_MYBIPEDAL Base Environment
######################


@configclass
class SF_MYBIPEDAL_BaseEnvCfg(SF_MYBIPEDAL_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SOLEFOOT_MYBIPEDAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.3, 0.3)

        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class SF_MYBIPEDAL_BaseEnvCfg_PLAY(SF_MYBIPEDAL_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


############################
# SF_MYBIPEDAL Blind Flat Environment
############################


@configclass
class SF_MYBIPEDAL_BlindFlatEnvCfg(SF_MYBIPEDAL_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class SF_MYBIPEDAL_BlindFlatEnvCfg_PLAY(SF_MYBIPEDAL_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

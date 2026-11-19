"""G1 29-DoF velocity task, tuned for a steadier (less "drunk") gait.

Same robot, observations and actions as velocity_env_cfg.RobotEnvCfg, so policies trained
here run unchanged in the same deploy/sim2sim scripts. Only rewards, command ranges and
curricula differ. Motivated by the 2026-11-18_16-33-49 run of the base task, whose final
metrics showed loose velocity tracking (xy error ~0.28 m/s) and a large body yaw-rate error
(~0.82 rad/s with commands of at most 0.2 rad/s), i.e. the torso twists/weaves while walking.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RobotEnvCfg


@configclass
class SteadyRobotEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # -- tighter linear velocity tracking (base: std 0.5). With std 0.5 a 0.28 m/s error still
        #    earns ~73% of the reward; with 0.25 it earns ~29%, so sloppy tracking stops paying.
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(0.0625)  # = 0.25

        # -- turning matters as much as walking (base: 0.5). Std left at 0.5: the current yaw error
        #    (~0.8 rad/s) is already far outside it, so tightening would only remove the gradient.
        self.rewards.track_ang_vel_z.weight = 1.0

        # -- penalize roll/pitch rocking of the torso harder (base: -0.05)
        self.rewards.base_angular_velocity.weight = -0.1

        # -- actually train turning: the base task's yaw range starts at +-0.1 rad/s and has no
        #    curriculum, so it never grows (play/deploy then use +-0.2, which it never saw).
        #    Enable the yaw curriculum and let it grow to +-0.5 rad/s.
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.5, 0.5)
        self.curriculum.ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)


@configclass
class SteadyRobotPlayEnvCfg(SteadyRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

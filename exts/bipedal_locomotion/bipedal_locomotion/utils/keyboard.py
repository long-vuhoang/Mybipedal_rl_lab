# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(2) velocity teleop (bipedal_locomotion / ManagerBasedRLEnv)."""

import weakref
from collections.abc import Callable

import carb
import omni
import torch
from isaaclab.devices.device_base import DeviceBase
from isaaclab.envs import ManagerBasedRLEnv


class Keyboard(DeviceBase):

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "base_velocity",
        lin_vel_step: float = 0.05,
        ang_vel_step: float = 0.05,
    ):
        """Initialize the keyboard layer.

        Args:
            env: The (possibly wrapped) environment to control. Only requires
                `.unwrapped` to expose `command_manager` and `episode_length_buf`.
            command_name: Name of the velocity command term in CommandsCfg
                (matches `base_velocity` in SF_MINIBOT_EnvCfg.CommandsCfg).
            lin_vel_step: Step size for linear velocity control.
            ang_vel_step: Step size for angular velocity control.
        """
        self.env = env
        self.command_name = command_name
        self.lin_vel_step = lin_vel_step
        self.ang_vel_step = ang_vel_step

        # velocity command state
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel = 0.0

        # sanity check: command term must exist and heading_command must be off,
        # otherwise the command_manager overwrites ang_vel_z every step from the
        # heading controller and keyboard yaw control (Q/E) has no effect.
        unwrapped = self.env.unwrapped
        term_cfg = unwrapped.command_manager.get_term(self.command_name).cfg
        if getattr(term_cfg, "heading_command", False):
            print(
                f"[Keyboard] WARNING: '{self.command_name}' has heading_command=True; "
                "ang_vel_z will be overwritten by the heading controller each step. "
                "Set env_cfg.commands.base_velocity.heading_command = False before "
                "creating the env if you want manual yaw control."
            )

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

        print("[Keyboard] Velocity control initialized:")
        print("  W/S: Forward/Backward")
        print("  A/D: Strafe left/right")
        print("  Q/E: Turn left/right")
        print("  X: Stop (zero velocity)")
        print("  R: Reset environment")

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for ManagerBasedRLEnv: {self.__class__.__name__}\n"
        return msg

    """
    Operations
    """

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        pass

    def advance(self):
        pass

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            if event.input.name in self._INPUT_KEY_MAPPING:
                key = event.input.name

                # Reset environment
                if key == "R":
                    env = self.env.unwrapped
                    env.episode_length_buf = torch.ones_like(env.episode_length_buf) * 1e6
                    print("[Keyboard] Environment reset triggered")
                # Forward/Backward (lin_vel_x)
                elif key == "W":
                    self.lin_vel_x += self.lin_vel_step
                    self._update_commands()
                elif key == "S":
                    self.lin_vel_x -= self.lin_vel_step
                    self._update_commands()
                # Turn left/right (ang_vel_z)
                elif key == "Q":
                    self.ang_vel += self.ang_vel_step
                    self._update_commands()
                elif key == "E":
                    self.ang_vel -= self.ang_vel_step
                    self._update_commands()
                # Strafe left/right (lin_vel_y)
                elif key == "A":
                    self.lin_vel_y += self.lin_vel_step
                    self._update_commands()
                elif key == "D":
                    self.lin_vel_y -= self.lin_vel_step
                    self._update_commands()
                # Stop (zero velocity)
                elif key == "X":
                    self.lin_vel_x = 0.0
                    self.lin_vel_y = 0.0
                    self.ang_vel = 0.0
                    self._update_commands()
                    print("[Keyboard] Stopped - all velocities set to zero")

        # since no error, we are fine :)
        return True

    def _update_commands(self):
        """Update the velocity command term through the command_manager."""
        env = self.env.unwrapped
        # get_command returns the live tensor (view), in-place edits persist
        command = env.command_manager.get_command(self.command_name)
        command[:, 0] = self.lin_vel_x  # lin_vel_x
        command[:, 1] = self.lin_vel_y  # lin_vel_y
        command[:, 2] = self.ang_vel    # ang_vel_z
        print(f"[Keyboard] Vel: vx={self.lin_vel_x:.2f}, vy={self.lin_vel_y:.2f}, ang_vel={self.ang_vel:.2f}")

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            "W": "forward",
            "S": "backward",
            "Q": "turn_left",
            "E": "turn_right",
            "A": "strafe_left",
            "D": "strafe_right",
            "X": "stop",
            "R": "reset_envs",
        }
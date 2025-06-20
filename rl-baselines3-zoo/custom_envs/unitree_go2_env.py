import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium.spaces import Box

class UnitreeGo2Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        model_path = os.path.join(os.path.dirname(__file__), "assets", "unitree_go2", "go2.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        obs_size = self.model.nq + self.model.nv
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Set robot to 'home' keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        obs = np.concatenate([self.data.qpos, self.data.qvel])

        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render()

        return obs, {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        reward = float(self.data.qpos[0])  # forward progress
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        terminated = False  # no termination
        truncated = False   # no truncation
        info = {}

        return obs, reward, terminated, truncated, info


    def render(self):
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium.spaces import Box

class UnitreeGo2Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        model_path = os.path.join(os.path.dirname(__file__), "assets", "unitree_go2", "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        obs_size = self.model.nq + self.model.nv
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        self.step_limit = 1000
        self.step_count = 0

        self.hip_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        ]

        self.feet_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["FL", "RL", "FR", "RR"]
        ]



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

        self.step_count = 0

        return obs, {}

    def step(self, action):

        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])

        self.data.ctrl[:] = action * 23.7
        mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        new_x = float(self.data.qpos[0])
        new_y = float(self.data.qpos[1])
        z = float(self.data.qpos[2])  # height of base
        forward_vel = float(self.data.qvel[0])  # x velocity
        forward_move = np.abs(new_x - x)
        reward = (100 * forward_move) + (np.clip(forward_vel, -0.5, 0.5))
        reward = np.clip(reward, -10, 10)

        hip_positions = [self.data.xpos[i] for i in self.hip_body_ids]
        FL_hip, FR_hip, RL_hip, RR_hip = hip_positions

        foot_positions = [self.data.geom_xpos[i] for i in self.feet_indices]
        FL_y = self.data.geom_xpos[self.feet_indices[0]][1]
        RL_y = self.data.geom_xpos[self.feet_indices[1]][1]
        FR_y = self.data.geom_xpos[self.feet_indices[2]][1]
        RR_y = self.data.geom_xpos[self.feet_indices[3]][1]

        out_of_line = (np.abs(new_y - y) >= 0.1 or
                      np.abs(FL_hip[1] - RL_hip[1]) > 0.1 or
                      np.abs(FR_hip[1] - RR_hip[1]) > 0.1 or
                      np.abs(FL_y - RL_y) > 0.1 or
                      np.abs(FR_y - RR_y) > 0.1)
        
        terminated = z <= 0.1
        
        if terminated:
            # if out_of_line:
            #     reward -= 40
            reward -= 10

        truncated = self.step_count >= self.step_limit
        info = {
            "x_position": float(self.data.qpos[0]),
            "x_velocity": float(self.data.qvel[0]),
            "reward_forward": reward
        }

        return obs, reward, terminated, truncated, info


    def render(self):
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

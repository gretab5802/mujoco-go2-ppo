import os
import gymnasium as gym
import mujoco.renderer
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium.spaces import Box
import math
from scipy.spatial.transform import Rotation as R
from mujoco import Renderer


class UnitreeGo2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        model_path = os.path.join(os.path.dirname(__file__), "assets", "unitree_go2", "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        obs_size = self.model.nq + self.model.nv
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        self.dt = self.model.opt.timestep
        self.step_limit = int(20.0 / self.dt)
        self.step_count = 0

        self.command_x = 0.5
        self.command_y = 0.0
        self.command_yaw = 0
        self.base_height_target = 0.3
        self.tracking_sigma = 0.25

        # Set physics parameters to stabilize simulation
        # self.model.dof_damping[6:] = 4.0
        # self.model.dof_frictionloss[6:] = 1.0
        # self.model.dof_armature[6:] = 0.05

        # Map joint names to indices
        self.joint_indices = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is not None:
                self.joint_indices[name] = i

        self.default_dof_pos = self._get_default_joint_pos()
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)

        self.logger = RewardLogger()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        y_before = self.data.qpos[1]
        x_before = self.data.qpos[0]

        self.step_count += 1

        ctrl_range = self.model.actuator_ctrlrange
        mid = 0.5 * (ctrl_range[:, 0] + ctrl_range[:, 1])
        amp = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        self.data.ctrl[:] = mid + action * amp


        mujoco.mj_step(self.model, self.data)

        y_after = self.data.qpos[1]
        x_after = self.data.qpos[0]

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_termination()
        truncated = self.step_count >= self.step_limit

        info = {
            "x_position": float(self.data.qpos[0]),
            "x_velocity": float(self.data.qvel[0]),
            "rew/vel":          self._reward_tracking_velocity(),
            "rew/survive":      self._reward_survival_bonus(),
            "rew/height":       self._penalty_height(),
            "rew/posture":      self._penalty_posture(),
            "rew/hip_limit":    self._penalty_hip_duction(),
        }

        self.logger.update({
            "velocity": self._reward_tracking_velocity(),
            "height": self._penalty_height(),
            "head_height": self._penalty_head_height(),
            "y": self._penalty_lateral(),
            "head_y": self._penalty_head_lateral(),
            "posture": self._penalty_posture(),
            "torque": self._penalty_torque_effort(),
            "pose_penalty": self._penalty_pose_different(),
            "ang_velocity": self._reward_tracking_ang_vel(),
            "action_rate": self._penalty_action_rate(),
            "hip_limit": self._penalty_hip_duction(),  # penalty for hip joints folding inwards
            "survival bonus": self._reward_survival_bonus()
        })

        self.last_action = np.copy(self.data.ctrl)

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.nan_to_num(np.concatenate([self.data.qpos, self.data.qvel]), nan=0.0)

    def _check_termination(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tracker")
        head_pos = np.copy(self.data.site_xpos[site_id])
        y = self.data.qpos[1]
        z = self.data.qpos[2]

        # Extract rotation matrix
        R = self.data.xmat[0].reshape(3, 3)

        # Estimate roll and pitch from rotation matrix
        pitch = np.arcsin(-R[2, 0])         # around y-axis
        roll = np.arctan2(R[2, 1], R[2, 2]) # around x-axis

        # Too much side-to-side motion
        if np.abs(y) > 0.2 or np.abs(head_pos[1]) > 0.2:
            return True
        # Body too low or too high
        if z < 0.2 or z > 0.4:
            return True
        # Too much body rolling/rotation
        if abs(roll) > 0.2 or abs(pitch) > 0.2:
            return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            if not hasattr(self, "_renderer"):
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data, camera="follow")
            return self._renderer.render()
        elif self.render_mode == "human" and self.viewer:
            self.viewer.sync()


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.logger.write_averages()

    def _get_default_joint_pos(self):
        default_angles = {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        }
        joint_pos = np.copy(self.data.qpos)
        for name, angle in default_angles.items():
            idx = self.joint_indices.get(name)
            if idx is not None and idx < len(joint_pos):
                joint_pos[idx] = angle
        return joint_pos[:self.model.nu]

    def _compute_reward(self, action):
        reward = (
            2 * self._reward_tracking_velocity()                # body moving forward
            - self._penalty_height()                            # penalize height deviation
            - self._penalty_head_height()                       # penalize height deviation
            - self._penalty_lateral()                           # penalize side-to-side body motion
            - self._penalty_head_lateral()                      # penalize side-to-side head motion
            - 10 * self._penalty_posture()                      # posture: penalize roll/pitch/yaw
            # - 0.000001 * self._penalty_torque_effort()        # penalize actuator strain
            - 0.05 * self._penalty_pose_different()             # pose penalty: penalize unnatural joint config
            + self._reward_tracking_ang_vel()                   # ang velocity close to target (0)
            - 0.00005 * self._penalty_action_rate()             # penalize changes in actions
            - 100 * self._penalty_hip_duction()                 # penalize hip joints folding inwards or outwards
            + self._reward_survival_bonus()                     # survival bonus
        )
        return reward

    def _reward_tracking_velocity(self):
        target = self.command_x  # 0.5 at the moment
        current = self.data.qvel[0]
        return np.exp(-((current - target) ** 2) / self.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = np.square(self.data.qvel[5] - self.command_yaw)
        return np.exp(-ang_vel_error / self.tracking_sigma)

    def _penalty_height(self):
        return (self.data.qpos[2] - 0.27) ** 2
        
    def _penalty_head_height(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tracker")
        head_pos = np.copy(self.data.site_xpos[site_id])
        return (head_pos[2] - 0.37) ** 2

    def _penalty_lateral(self):
        return (np.abs(self.data.qpos[1]))

    def _penalty_head_lateral(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tracker")
        head_pos = np.copy(self.data.site_xpos[site_id])
        return (np.abs(head_pos[1]))

    def _penalty_posture(self):
        quat = self.data.qpos[3:7]
        rpy = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz", degrees=False)
        return rpy[0] ** 2 + rpy[1] ** 2 + rpy[2] ** 2

    def _penalty_torque_effort(self):
        return np.sum(np.square(self.data.ctrl))

    def _penalty_action_rate(self):
        return np.sum(np.square(self.last_action - self.data.ctrl))

    def _reward_survival_bonus(self):
        return 1.0

    def _penalty_hip_duction(self):
        """Penalize hip joints by how far they exceed the abduction/adduction limits (summed absolute violation)."""
        hip_joint_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        hip_joint_indices = [self.joint_indices[name] for name in hip_joint_names]
        hip_angles = self.data.qpos[hip_joint_indices]
        # Positive amount by which each joint exceeds the limits
        violation = hip_angles ** 2
        return np.sum(violation) * 10.0  # scale factor

    # def _penalty_pose_different(self):
    #     # abduction, thigh joint, calf joint
    #     weights = [2, 1, 1]
    #     joint_pos = self.data.qpos[7:7 + self.model.nu]
    #     default_joint_pos = np.array([
    #         0, 0.9, -1.8,  # FL
    #         0, 0.9, -1.8,  # FR
    #         0, 0.9, -1.8,  # RL
    #         0, 0.9, -1.8   # RR
    #     ])
    #     abduction = np.sum(np.abs(joint_pos[[0, 3, 6, 9]] - default_joint_pos[[0, 3, 6, 9]]))
    #     thigh = np.sum(np.abs(joint_pos[[1, 4, 5, 10]] - default_joint_pos[[1, 4, 5, 10]]))
    #     calf = np.sum(np.abs(joint_pos[[2, 5, 8, 11]] - default_joint_pos[[2, 5, 8, 11]]))
    #     return np.dot(weights, [abduction, thigh, calf])

    def _penalty_pose_different(self):
        joint_pos = self.data.qpos[7:7 + self.model.nu]
        default_joint_pos = np.array([
            0, 0.9, -1.8,  # FL
            0, 0.9, -1.8,  # FR
            0, 0.9, -1.8,  # RL
            0, 0.9, -1.8   # RR
        ])
        return np.sum(np.abs(joint_pos - default_joint_pos))


class RewardLogger:
    def __init__(self):
        self.count = 0
        self.totals = {
            "velocity": 0.0,
            "height": 0.0,
            "head_height": 0.0,
            "y": 0.0,
            "head_y": 0.0,
            "posture": 0.0,
            "torque": 0.0,
            "pose_penalty": 0.0,
            "ang_velocity": 0.0,
            "action_rate": 0.0,
            "hip_limit": 0.0,
            "survival bonus": 0.0
        }

    def update(self, values):
        self.count += 1
        for key in self.totals:
            self.totals[key] += values[key]

    def write_averages(self, filename="reward_log.txt"):
        if self.count == 0:
            return
        with open(filename, "w") as f:
            f.write(f"Averages over {self.count} steps:\n")
            for key, total in self.totals.items():
                f.write(f"{key}: {total / self.count:.6f}\n")

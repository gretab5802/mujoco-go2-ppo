from stable_baselines3 import PPO
from custom_envs.unitree_go2_env import UnitreeGo2Env
import time
import numpy as np
import os

os.environ["MUJOCO_GL"] = "glfw"

env = UnitreeGo2Env(render_mode="human")
model = PPO.load("logs/ppo/UnitreeGo2-v0_8/UnitreeGo2-v0.zip")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, _, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()
    time.sleep(0.01)

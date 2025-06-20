from custom_envs.unitree_go2_env import UnitreeGo2Env

env = UnitreeGo2Env()
env.reset()
env.render()

# keep the viewer open
while True:
    env.step(env.action_space.sample())

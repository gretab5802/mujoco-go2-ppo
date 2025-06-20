# Import PPO algorithm from Stable-Baselines3
from stable_baselines3 import PPO

# Import Gymnasium for the simulation environment
import gymnasium as gym

# STEP 1: Create the MuJoCo environment
# "HalfCheetah-v4" is a standard continuous control benchmark
# No "render_mode" specified, so it runs headless (no GUI)
env = gym.make("HalfCheetah-v4")

# STEP 2: Initialize the PPO model
# "MlpPolicy" uses a multi-layer perceptron (fully connected NN)
# "env" is the environment we want to train in
# "verbose=1" prints training logs to the console
model = PPO("MlpPolicy", env, verbose=1)

# STEP 3: Train the model
# This will train the agent for however many timesteps specified
model.learn(total_timesteps=1_000_000)

# STEP 4: Save the trained model to a file
# This allows you to load it later for evaluation or deployment
model.save("ppo_halfcheetah")

# STEP 5: Clean up the environment
env.close()
# Unitree Go2 Locomotion in MuJoCo with PPO
Utilizing stable-baslines3 PPO to train Unitree Go2 quadruped forward walking locomotion in MuJoCo simulation. This was ran in a Docker container for Ubuntu 20.04 for compatability with ROS2 Foxy since that was the version of ROS2 on our real Go2 to prepare for future work. **This respository does not work with real robots at all, only training in simulation.**

## Resources

- [MuJoCo official site](https://mujoco.org/)  
- [Stable-Baselines3 GitHub](https://github.com/DLR-RM/stable-baselines3)  
- [RL-Zoo3 GitHub](https://github.com/DLR-RM/rl-zoo3)
### Helpful repositories, used for reward shaping ideas:
- [Genesis simulation with a Go2 (GitHub for their Go2 environment)](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/locomotion/go2_env.py)
- [Training Quadruped Locomotion using Reinforcement Learning in Mujoco (GitHub for training Go1 in MuJoCo)](https://github.com/nimazareian/quadruped-rl-locomotion/tree/main)

## Prerequisites

### Software
- Docker is optional, but since that is what I used (local machine has Ubuntu 24.04, used Docker for Ubuntu 20.04 workspace), some things in here will be particular to that. This tutorial does not go through installing Docker on Ubuntu. If you do not have it already, [see this for installing it on Ubuntu 22.04 or 24.04](https://docs.docker.com/engine/install/ubuntu/)
  
## Highlights of repository:
`rl-baselines3-zoo/custom_envs` contains all things particular to Go2, including go2.xml which gives us the go2 information and scene.xml which loads in the go2 already, as well as a checkered floor, located in `rl-baselines3-zoo/custom_envs/assets/unitree_go2` <br /><br />
`rl-baselines3-zoo/hyperparams/` contains all hyperparameter files for each algorithm, `ppo.yml` and `sac.yml` have a UnitreeGo2-v0 env which I added and registered in `rl-baselines3-zoo/rl_zoo3/import_envs.py`. The Unitree environment is defined in `rl-baselines3-zoo/custom_envs/unitree_go2_env.py`. **If you want to use a different algorithm or have a different custom environment, you must add hyperparameters to the relevant yml files in this folder** <br /><br />
`rl-baselines3-zoo/custom_envs/unitree_go2_env.py` is the most important file, it contains all of the reward shaping. **If you want to edit the rewards, edit this file. If you want to create your own environment, create a new file and register it in the import_envs.py file described above** <br />

## To train and visualize
Cd into the rl-baselines3-zoo repository
```
cd rl-baselines3-zoo/
```
To train, there are a lot of flags you can utilize, but these are the ones I stuck with. You can specify things like number of environments in the command to run training, but I already defined this in the yml hyperparameter files for PPO and SAC. <br /><br />
In general, the structure is: **python train.py --algo \[ALGORITHM] --env \[ENVIRONMENT] -f \[FOLDER TO STORE TRAINED POLICY] <br /><br />**
This is what mine usually looked like:
```
python train.py --algo ppo --env UnitreeGo2-v0 -f logs/
```
To visualize, it's the same command but you train `train.py` to `enjoy.py`:
```
python enjoy.py --algo ppo --env UnitreeGo2-v0 -f logs/
```
Note: -f logs/ will make it save automatically to logs/\[ALGO NAME], e.g. logs/ppo. For organization, I sometimes customized it to -f vel_survive/ to save the logs when I was only rewarding forward velocity and a survival bonus, all the way up to folders like vel_survive_drift_ang_pose ... and so on. It made for very long file names but it made it easier for data analysis/comparison and future ablation studies with the reward terms.

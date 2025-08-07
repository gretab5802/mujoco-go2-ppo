# Training a Unitree Go2 Quadruped locomotion in MuJoCo using PPO
Utilizing stable-baslines3 to train Unitree Go2 quadruped forward walking locomotion in MuJoCo simulation. This was ran in a Docker container for Ubuntu 20.04 for compatability with ROS2 Foxy since that was the version of ROS2 our Go2 had.

## Resources

- **MuJoCo**  
  – [Official site](https://mujoco.org/)  
- **Stable-Baselines3 & RL-Zoo3**  
  – [Stable-Baselines3 GitHub](https://github.com/DLR-RM/stable-baselines3)  
  – [RL-Zoo3 GitHub](https://github.com/DLR-RM/rl-zoo3)  
- **Unitree Go2 ROS 2 Drivers**  
  – [go2_control_interface](https://github.com/inria-paris-robotics-lab/go2_control_interface)  
  – [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)  
- **ROS 2 Foxy**  
  – [Installation guide](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

## Prerequisites

### Hardware
- The beefier the GPUs, the better

### Software
- Docker is ultimately optional, but since that is what I used (local machine has Ubuntu 24.04, used Docker for Ubuntu 20.04 workspace), some things in here will be particular to that. This tutorial does not go through installing Docker on Ubuntu. If you do not have it already, [see this for installing it on Ubuntu 22.04 or 24.04](https://docs.docker.com/engine/install/ubuntu/)
  
# Highlights of repository:
`rl-baselines3-zoo/custom_envs` contains all things particular to Go2, including go2.xml which gives us the go2 information and scene.xml which loads in the go2 already, as well as a checkered floor, located in `rl-baselines3-zoo/custom_envs/assets/unitree_go2` <br />

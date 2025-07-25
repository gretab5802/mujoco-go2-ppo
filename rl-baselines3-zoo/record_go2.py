#!/usr/bin/env python3
"""Record a trained SB3 model controlling UnitreeGo2Env into an MP4.  
Usage example:
    python record_go2.py \
        --algo ppo \
        --model logs/ppo/UnitreeGo2-v0_8/UnitreeGo2-v0.zip \
        --video-folder videos/run8 \
        --episodes 3
"""
import argparse
from pathlib import Path

from rl_zoo3.import_envs import UnitreeGo2Env
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit

from stable_baselines3 import PPO, SAC


ALGOS = {
    "ppo": PPO,
    "sac": SAC
}

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record UnitreeGo2 agent into video")
    parser.add_argument("--algo", required=True, choices=ALGOS.keys(), help="RL algorithm")
    parser.add_argument("--model", required=True, type=str, help="Path to .zip model file")
    parser.add_argument("--video-folder", required=True, type=str, help="Output directory for mp4 files")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device")
    return parser

def main():
    args = make_parser().parse_args()
    algo_cls = ALGOS[args.algo]
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file '{model_path}' not found")

    video_dir = Path(args.video_folder)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create env with rgb_array rendering so RecordVideo can capture frames
    base_env = gym.make("UnitreeGo2-v0", render_mode="rgb_array")

    # stop each episode after 600 simulation steps (~20 s at your dt)
    base_env = TimeLimit(base_env, max_episode_steps=600)

    env = RecordVideo(
        base_env,
        video_folder=str(video_dir),
        episode_trigger=lambda idx: True,     # every episode
        name_prefix=model_path.stem,
    )

    model = algo_cls.load(str(model_path), env=env, device=args.device)

    for ep in range(args.episodes):
        done, truncated = False, False
        obs, _ = env.reset()
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
    env.close()
    print(f"Videos written to {video_dir.resolve()}")

if __name__ == "__main__":
    main() 

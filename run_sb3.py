import sys

# import gym
import gymnasium as gym

import time

from stable_baselines3 import SAC
from sb3_contrib import SACD

import sys

sys.path.append("./highway-env/")
import highway_env

if __name__ == "__main__":
    env_name = "MountainCarContinuous-v0"
    # model_prefix = "model"
    model_prefix = "MountainCarContinuous-v0"
    if len(sys.argv) < 3:
        print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix>")
        print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
    else:
        env_name = sys.argv[1]
        model_prefix = sys.argv[2]
    model_save_file = model_prefix + ".zip"

    env = gym.make(env_name)  # , render_mode="human")
    obs, _ = env.reset()

    if "merge" in env_name:
        model = SACD.load(model_save_file, env)
    else:
        model = SAC.load(model_save_file, env)

    for i in range(100000):
        start = time.time()
        action, _state = model.predict(obs, deterministic=True)
        end = time.time()
        print(f"Inference took {(end-start)*1000}ms")
        obs, reward, done, _, info = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()

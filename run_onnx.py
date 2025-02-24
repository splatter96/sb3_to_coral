import sys

# import gym
import gymnasium as gym
import onnxruntime as ort
import numpy as np
import time

sys.path.append("./highway-env/")
import highway_env

if __name__ == "__main__":
    env_name = "MountainCarContinuous-v0"
    # model_prefix = 'model'
    model_prefix = "MountainCarContinuous-v0"
    if len(sys.argv) < 3:
        print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix>")
        print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
    else:
        env_name = sys.argv[1]
        model_prefix = sys.argv[2]
    model_save_file = model_prefix + ".onnx"

    env = gym.make(env_name)  # , render_mode="human")
    obs, _ = env.reset()
    # obs = np.array(obs)

    ort_session = ort.InferenceSession(model_save_file)

    for i in range(100000):
        # print(f"step {i}")
        # print(f"{obs=}")
        # print(obs.dtype)
        # print(obs.reshape([1, -1]).astype(np.float32).dtype)
        start = time.time()
        outputs = ort_session.run(
            None, {"input": obs.reshape([1, -1]).astype(np.float32)}
        )
        end = time.time()
        print(f"Inference took {(end-start)*1000}ms")
        # print(f"{outputs=}")
        obs, reward, done, _, info = env.step(outputs[0])
        env.render()
        if done:
            obs, _ = env.reset()

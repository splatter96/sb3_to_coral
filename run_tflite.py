import sys
import numpy as np

# import gym
import gymnasium as gym
import tflite_runtime.interpreter as tflite
import time

sys.path.append("./highway-env/")
import highway_env

if __name__ == "__main__":
    env_name = "MountainCarContinuous-v0"
    # model_prefix = 'model_quant'
    # model_prefix = "MountainCarContinuous-v0_quant"
    # model_prefix = "MountainCarContinuous-v0"
    model_prefix = "MountainCarContinuous-v0_quant_edgetpu"
    if len(sys.argv) < 3:
        print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix>")
        print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
    else:
        env_name = sys.argv[1]
        model_prefix = sys.argv[2]
    model_save_file = model_prefix + ".tflite"

    delegates = None
    if "edgetpu" in model_save_file:
        delegates = [tflite.load_delegate("libedgetpu.so.1")]

    env = gym.make(env_name)  # , render_mode="human")
    obs, _ = env.reset()

    interpreter = tflite.Interpreter(
        model_path=model_save_file, experimental_delegates=delegates
    )
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(100000):
        start = time.time()
        input_data = obs.reshape(1, -1).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        end = time.time()
        print(f"Inference took {(end-start)*1000}ms")

        obs, reward, done, _, info = env.step(output_data)
        env.render()
        if done:
            obs, _ = env.reset()

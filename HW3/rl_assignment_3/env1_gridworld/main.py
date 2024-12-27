import json
import os
import warnings
from time import time

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO, SAC
from PIL import Image

from gridworld import GridWorld

warnings.filterwarnings("ignore")

STEP_REWARD = -0.01
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0
EXIT_REWARD = 0.1
BAIT_REWARD = 1.0
BAIT_STEP_PENALTY = -0.25
MAX_STEP = 1000
RENDER_MODE = "ansi"


def print_state_action(obs: np.ndarray, action: np.ndarray, env: gym.Env) -> None:
    """Print the state and action taken.

    Args:
        obs (np.ndarray): The observation from the environment.
        action (np.ndarray): The action taken.
        env (gym.Env): The environment instance.
    """
    print("state:", obs[0], "action: ", env.grid_world.ACTION_INDEX_TO_STR[action[0]])


def test_correctness(filename: str = "tasks/maze.txt") -> None:
    """Test the correctness of the task based on the ground truth trajectory .

    Args:
        filename (str): The path to the maze file.
    """
    # Extract the task name from the filename
    task_name = os.path.split(filename)[1].replace(".txt", "")
    if "_" in task_name:
        task_name = task_name.split('_')[0]
    grid_world = GridWorld(
        maze_file=filename,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        exit_reward=EXIT_REWARD,
        bait_reward=BAIT_REWARD,
        bait_step_penalty=BAIT_STEP_PENALTY,
        max_step=MAX_STEP,
    )

    # Load ground truth trajectory from csv file
    df = pd.read_csv(os.path.join("grading", f"gridworld_{task_name}.csv"))
    state = df["state"]
    action = df["action"]
    reward = df["reward"]
    done = df["done"]
    truncated = df["truncated"]
    next_state = df["next_state"]
    
    result = []
    grid_world.set_current_state(state[0])

    # Iterate through the ground truth data
    for _, a, r, d, t, ns in zip(state, action, reward, done, truncated, next_state):
        next_state_prediction, reward_prediction, done_prediction, truncated_prediction = grid_world.step(a)
        
        # Check if the model prediction matches groud truth
        if done_prediction:
            next_state_prediction = grid_world.reset()
            result.append( (next_state_prediction in grid_world._init_states) and reward_prediction == r and done_prediction == d and truncated_prediction == t)
            grid_world.set_current_state(ns)
        else:
            result.append(next_state_prediction == ns and reward_prediction == r and done_prediction == d and truncated_prediction == t)

    print(f"The correctness of the task {task_name}: {np.round(np.mean(result) * 100, 2)} %")


def write_gif(filename: str = "lava.txt", algorithm: type = PPO) -> None:
    """Writes the trajectory of the task to a GIF file.

    Args:
        filename (str): The path to the task file.
        algorithm (type): The algorithm to use (default is PPO).
    """
    # Extract the task name from the filename
    task_name = os.path.split(filename)[1].replace(".txt", "")
    
    # Register the env and make env
    gym.register(f"GridWorld{task_name.capitalize()}-v1", entry_point="gridworld:GridWorldEnv")
    env = gym.make(
        f"GridWorld{task_name.capitalize()}-v1",
        render_mode=RENDER_MODE,
        maze_file=filename,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        exit_reward=EXIT_REWARD,
        bait_reward=BAIT_REWARD,
        bait_step_penalty=BAIT_STEP_PENALTY,
        max_step=MAX_STEP,
    )

    # Visualize and load model
    env.grid_world.visualize(f"{task_name}.png")
    model = algorithm.load(f"assets/gridworld_{task_name}", env=env)
    vec_env = model.get_env()
    obs = vec_env.reset()

    # Render every step and store the RGB frames for the GIF
    states = []
    while True:
        action, _states = model.predict(obs, deterministic=True)
        # You can use print state action to debug
        # print_state_action(obs, action, env)

        rgb = env.grid_world.get_rgb()
        states.append(rgb.copy())
        obs, reward, done, info = vec_env.step(action)
        if done:
            break

    # Create and save the GIF from the collected frames
    images = [Image.fromarray(state) for state in states]
    images = iter(images)
    image = next(images)
    image.save(
        f"gif/gridworld_{task_name}.gif",
        format="GIF",
        save_all=True,
        append_images=images,
        loop=0,
        fps=1,
    )


if __name__ == "__main__":
    # TEST CORRECTNESS
    test_correctness("tasks/lava.txt")
    test_correctness("tasks/exit.txt")
    test_correctness("tasks/bait.txt")
    test_correctness("tasks/door.txt")
    test_correctness("tasks/portal.txt")
    test_correctness("tasks/maze.txt")
    # 
    # Write one trajectory to gif
    # write_gif("tasks/lava.txt", algorithm=PPO)
    # write_gif("tasks/exit.txt", algorithm=PPO)
    # write_gif("tasks/bait.txt", algorithm=PPO)
    # write_gif("tasks/door.txt", algorithm=A2C)
    # write_gif("tasks/portal.txt", algorithm=PPO)
    # write_gif("tasks/maze.txt", algorithm=PPO)

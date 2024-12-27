import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN

import numpy as np
from collections import Counter
import os
import re

register(
    id='2048-eval',
    entry_point='envs:Eval2048Env'
)

def evaluation(env, model, render_last, eval_num=100):
    """We only evaluate seeds 0-99 as our public test cases."""
    score = []
    highest = []

    ### Run eval_num times rollouts,
    for seed in range(eval_num):
        done = False
        # Set seed and reset env using Gymnasium API
        obs, info = env.reset(seed=seed)

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

        # Render the last board state of each episode
        # print("Last board state:")
        # env.render()

        score.append(info['score'])
        highest.append(info['highest'])

    ### Render last rollout
    if render_last:
        print("Rendering last rollout")
        done = False
        obs, info = env.reset(seed=eval_num-1)
        env.render()

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            env.render()

        
    return score, highest

def get_latest_model(directory):
    files = os.listdir(directory)
    model_numbers = [int(re.findall(r'\d+', file)[0]) for file in files if re.findall(r'\d+', file)]

    if model_numbers:
        latest_index = max(model_numbers)
        return os.path.join(directory, str(latest_index))
    else:
        raise FileNotFoundError(f"No models found in {directory}")

if __name__ == "__main__":
    model_path = get_latest_model("models/sample_model") # Change path name to load different models
    #model_path = "models/sample_model/1825"
    env = gym.make('2048-eval')

    ### Load model with SB3
    # Note: Model can be loaded with arbitrary algorithm class for evaluation
    # (You don't necessarily need to use PPO for training)
    model = DQN.load(model_path)

    eval_num = 100
    score, highest = evaluation(env, model, True, eval_num)

    print("Avg_score:  ", np.sum(score)/eval_num)
    print("Avg_highest:", np.sum(highest)/eval_num)


    print(f"Counts: (Total of {eval_num} rollouts)")
    c = Counter(highest)
    for item in (sorted(c.items(),key = lambda i: i[0])):
        print(f"{item[0]}: {item[1]}")

import numpy as np

from DP_solver import (
    IterativePolicyEvaluation,
    PolicyIteration,
    ValueIteration,
    AsyncDynamicProgramming,
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

STEP_REWARD = -1.0
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0
DISCOUNT_FACTOR = 0.9


def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt"):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def run_policy_evaluation(grid_world: GridWorld):
    print(bold(underline("Iterative Policy Evaluation")))
    policy = np.ones((grid_world.get_state_space(), 4)) / 4

    iterative_policy_evaluation = IterativePolicyEvaluation(
        grid_world, policy, discount_factor=DISCOUNT_FACTOR
    )
    iterative_policy_evaluation.run()

    grid_world.visualize(
        iterative_policy_evaluation.values,
        title=f"Iterative Policy Evaluation",
        show=False,
        filename=f"iterative_policy_evaluation.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    grid_world.reset()
    print()


def run_policy_iteration(grid_world: GridWorld):
    print(bold(underline("Policy Iteration")))
    policy_iteration = PolicyIteration(grid_world, discount_factor=DISCOUNT_FACTOR)
    policy_iteration.run()
    grid_world.visualize(
        policy_iteration.values,
        policy_iteration.policy,
        title=f"Policy Iteration",
        show=False,
        filename=f"policy_iteration.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(policy_iteration.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_value_iteration(grid_world: GridWorld):
    print(bold(underline("Value Iteration")))
    value_iteration = ValueIteration(grid_world, discount_factor=DISCOUNT_FACTOR)
    value_iteration.run()
    grid_world.visualize(
        value_iteration.values,
        value_iteration.policy,
        title=f"Value Iteration",
        show=False,
        filename=f"value_iteration.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(value_iteration.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_async_dynamic_programming(grid_world: GridWorld):
    print(bold(underline("Async Dynamic Programming")))
    async_dynamic_programming = AsyncDynamicProgramming(
        grid_world, discount_factor=DISCOUNT_FACTOR
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title=f"Async Dynamic Programming",
        show=False,
        filename=f"async_dynamic_programming.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


if __name__ == "__main__":
    grid_world = init_grid_world()
    run_policy_evaluation(grid_world)
    run_policy_iteration(grid_world)
    run_value_iteration(grid_world)
    run_async_dynamic_programming(grid_world)

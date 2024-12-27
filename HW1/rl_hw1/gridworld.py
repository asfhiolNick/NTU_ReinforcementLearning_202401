from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


class GridWorld:
    """Grid World"""

    ACTION_INDEX_TO_STR = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }
    ACTION_INDEX_TO_CHAR = {
        0: "^",
        1: "v",
        2: "<",
        3: ">",
    }
    OBJECT_INDEX_TO_STR = {
        0: "EMPTY",
        1: "WALL",
        2: "GOAL",
        3: "TRAP",
    }
    OBJECT_INDEX_TO_CHAR = {
        0: " ",
        1: "#",
        2: "G",
        3: "T",
    }

    def __init__(
        self,
        maze_file: str,
        goal_reward: float = 1,
        trap_reward: float = -1,
        step_reward: float = -1,
    ):
        """Constructor for GridWorld

        Args:
            maze_file (str): Path to the maze file
            goal_reward (float, optional): Reward in the goal state. Defaults to 1.
            trap_reward (float, optional): Reward in the trap state. Defaults to -1.
            step_reward (float, optional): Reward in the step state. Defaults to -1.
        """
        self.__goal_reward = goal_reward
        self.__trap_reward = trap_reward
        self.__step_reward = step_reward
        self.__step_count = 0
        self.__maze = np.array([])
        self.__state_list = []
        self.__read_maze(maze_file)

    def __read_maze(self, maze_file: str) -> None:
        """Read the maze file

        Returns:
            np.ndarray: Maze
        """
        self.__maze = np.loadtxt(maze_file, dtype=np.uint8)
        for i in range(self.__maze.shape[0]):
            for j in range(self.__maze.shape[1]):
                if self.__maze[i, j] != 1:
                    self.__state_list.append((i, j))

    def print_maze(self) -> None:
        """Print the maze"""
        print(f"Size: {self.__maze.shape}")
        for i in range(self.__maze.shape[0]):
            for j in range(self.__maze.shape[1]):
                print(self.OBJECT_INDEX_TO_CHAR[self.__maze[i, j]], end="")
            print()

    def get_step_count(self) -> int:
        """Return the step count

        Returns:
            int
        """
        return self.__step_count

    def get_action_space(self) -> int:
        """Return the action space

        Returns:
            int
        """
        return 4

    def get_state_space(self) -> int:
        """Return the state space

        Returns:
            int
        """
        return len(self.__state_list)

    def __is_valid_state(self, state_coord: tuple) -> bool:
        """Check if the state is valid (within the maze and not a wall)

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        if state_coord[0] < 0 or state_coord[0] >= self.__maze.shape[0]:
            return False
        if state_coord[1] < 0 or state_coord[1] >= self.__maze.shape[1]:
            return False
        if self.__maze[state_coord[0], state_coord[1]] == 1:
            return False
        return True

    def __is_goal_state(self, state_coord: tuple) -> bool:
        """Check if the state is a goal state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self.__maze[state_coord[0], state_coord[1]] == 2

    def __is_trap_state(self, state_coord: tuple) -> bool:
        """Check if the state is a trap state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self.__maze[state_coord[0], state_coord[1]] == 3

    def __get_next_state(self, state_coord: tuple, action: int) -> tuple:
        """Get the next state given the current state and action

        Args:
            state_coord (tuple)
            action (Action)

        Returns:
            tuple: next_state_coord
        """
        next_state_coord = np.array(state_coord)
        if action == 0:
            next_state_coord[0] -= 1
        elif action == 1:
            next_state_coord[0] += 1
        elif action == 2:
            next_state_coord[1] -= 1
        elif action == 3:
            next_state_coord[1] += 1
        if not self.__is_valid_state(next_state_coord):
            next_state_coord = state_coord
        return tuple(next_state_coord)

    def step(self, state: int, action: int) -> tuple:
        """Take a step in the environment

        Args:
            state (int)
            action (int)

        Returns:
            tuple: next_state, reward, done
        """
        self.__step_count += 1

        state_coord = self.__state_list[state]
        if self.__is_goal_state(state_coord):
            return state, self.__goal_reward, True
        if self.__is_trap_state(state_coord):
            return state, self.__trap_reward, True

        next_state_coord = self.__get_next_state(state_coord, action)
        next_state = self.__state_list.index(next_state_coord)

        return next_state, self.__step_reward, False

    def reset(self) -> None:
        """Reset the step count"""
        self.__step_count = 0

    def run_policy(
        self,
        policy: np.ndarray,
        start_state: int,
        max_steps: int = 1000,
    ) -> list:
        """Run the policy

        Args:
            policy (np.ndarray): Policy to run
            start_state (tuple): Start state
            max_steps (int, optional): Max steps to terminate. Defaults to 1000.

        Returns:
            list: History of states and actions
        """
        state = start_state
        history = []
        while True:
            action = policy[state]
            history.append((state, action))
            next_state, _, done = self.step(state, action)
            if done:
                history.append((next_state, None))
                break
            if len(history) > max_steps:
                print("Max steps reached.")
                break
            state = next_state
        return history

    def visualize(
        self,
        values: np.ndarray | None = None,
        policy: np.ndarray | None = None,
        title: str | None = None,
        show: bool = True,
        filename: str | None = "maze.png",
    ) -> None:
        """Visualize the maze

        Args:
            values (np.ndarray | None, optional): Values. Defaults to None.
            policy (np.ndarray | None, optional): Policy. Defaults to None.
            title (str | None, optional): Title. Defaults to None.
            show (bool, optional): Show the plot. Defaults to True.
            filename (str | None, optional): Filename to save. Defaults to "maze.png".
        """
        plt.close("all")
        # Empty: 0, Wall: 1, Goal: 2, Pit: 3
        cmap = colors.ListedColormap(["white", "black", "green", "red"])
        # adjust the size of the figure according to the maze
        fig, ax = plt.subplots(figsize=(self.__maze.shape[1], self.__maze.shape[0]))
        ax.imshow(self.__maze, cmap=cmap, vmin=0, vmax=4)
        ax.grid(which="major", axis="both", linestyle="-", color="gray", linewidth=2)
        ax.set_xticks(np.arange(-0.5, self.__maze.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, self.__maze.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        for i in range(self.__maze.shape[0]):
            for j in range(self.__maze.shape[1]):
                if self.__maze[i, j] == 1:
                    continue

                state = self.__state_list.index((i, j))
                label = f"{state}"
                if values is not None:
                    label += f"\n{values[state]:.4f}"
                if policy is not None and (self.__maze[i, j] == 0):
                    label += f"\n{self.ACTION_INDEX_TO_CHAR[policy[state]]}"

                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color="k",
                )
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()


if __name__ == "__main__":
    maze_file = "maze.txt"
    grid_world = GridWorld(maze_file)
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)

    start = 0
    next_state, reward, done = grid_world.step(start, 3)
    print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")

    print(f"Step count: {grid_world.get_step_count()}")

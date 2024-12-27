from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib import colors

COLORS = [
    "white",
    "black",
    "green",
    "red",
    "darkorange",
    "springgreen",
    "yellow",
    "brown",
    "aquamarine",
    "skyblue"
]


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
    OBJECT_TO_INDEX = {
        "EMPTY": 0,
        "WALL": 1,
        "GOAL": 2,
        "TRAP": 3,
        "LAVA": 4,
        "EXIT": 5,
        "KEY": 6,
        "DOOR": 7,
        "BAIT": 8,
        "PORTAL": 9,
        "AGENT": 10,
    }
    OBJECT_INDEX_TO_CHAR = {
        0: " ",
        1: "#",
        2: "G",
        3: "T",
        4: "L",
        5: "E",
        6: "K",
        7: "D",
        8: "B",
        9: "P",
        10: "A",
    }

    def __init__(
        self,
        maze_file: str,
        goal_reward: float = 1,
        trap_reward: float = -1,
        step_reward: float = -1,
        exit_reward: float = 0.1,
        bait_reward: float = 1,
        bait_step_penalty: float = -0.25,
        max_step: int = 1000,
    ):
        """Constructor for GridWorld

        Args:
            maze_file (str): Path to the maze file
            goal_reward (float, optional): Reward in the goal state. Defaults to 1.
            trap_reward (float, optional): Reward in the trap state. Defaults to -1.
            step_reward (float, optional): Reward in the step state. Defaults to -1.
            exit_reward (float, optional): Reward in the exit state. Defaults to 0.1.
            bait_reward (float, optional): Reward in the bait state. Defaults to 1.
            bait_step_penalty (float, optional): Penalty in the bait state. Defaults to -0.25.
            max_step (int, optional): Maximum number of steps. Defaults to 1000.
        """
        self._goal_reward = goal_reward
        self._trap_reward = trap_reward
        self._step_reward = step_reward
        self._exit_reward = exit_reward
        self._bait_reward = bait_reward
        self._bait_step_penalty = bait_step_penalty
        self.step_reward = self._step_reward
        self._step_count = 0
        self._maze = np.array([])
        self._state_list = []
        self._current_state = 0
        self.max_step = max_step
        self.maze_name = maze_file.split('/')[-1].replace(".txt", "").capitalize()
        self._read_maze(maze_file)
        self.render_init(self.maze_name)

        # if min_y is None you can initialize the agent in any state
        # if min_y is not None, you can initialize the agent in the state left to min_y
        min_y = None

        # lava init
        lava_states = []
        for state in range(self.get_grid_space()):
            if self._is_lava_state(self._state_list[state]):
                lava_states.append(self._state_list[state])

        if len(lava_states) > 0:
            # get the leftest coordinate of lava states
            min_y = min(lava_states, key=lambda x: x[1])[1]

        # record the door state and key state
        self._door_state = None
        self._key_state = None
        for state in range(self.get_grid_space()):
            if self._is_door_state(self._state_list[state]):
                self._door_state = state
            if self._is_key_state(self._state_list[state]):
                self._key_state = state

        # the door state and key state should be different and unique
        if self._door_state is None:
            # if there is no door state, then there should be no key state
            assert self._key_state is None
        else:
            # if there is door state, then there should be key state
            assert self._key_state is not None
            assert self._door_state != self._key_state

            if min_y is not None:
                min_y = min(min_y, self._state_list[self._door_state][1])
            else:
                min_y = self._state_list[self._door_state][1]

        self._bait_state = None
        for state in range(self.get_grid_space()):
            if self._is_bait_state(self._state_list[state]):
                self._bait_state = state

        self._portal_state = []
        for state in range(self.get_grid_space()):
            if self._is_portal_state(self._state_list[state]):
                self._portal_state.append(state)

        # only 2 portal or zero portal is valid
        assert len(self._portal_state) == 2 or len(self._portal_state) == 0
        self.portal_next_state = {}
        if len(self._portal_state) == 2:
            self.portal_next_state[self._state_list[self._portal_state[0]]
                                   ] = self._state_list[self._portal_state[1]]
            self.portal_next_state[self._state_list[self._portal_state[1]]
                                   ] = self._state_list[self._portal_state[0]]

        self._init_states = []
        for state in range(self.get_grid_space()):
            if state == self._bait_state:
                continue
            if state == self._key_state:
                continue
            if min_y is not None and self._state_list[state][1] < min_y:
                self._init_states.append(state)
            elif min_y is None:
                self._init_states.append(state)

        assert len(self._init_states) > 0
        self.reset()

    def _read_maze(self, maze_file: str) -> None:
        """Read the maze file

        Returns:
            np.ndarray: Maze
        """
        self._maze = np.loadtxt(maze_file, dtype=np.uint8)
        for i in range(self._maze.shape[0]):
            for j in range(self._maze.shape[1]):
                if self._maze[i, j] != 1:
                    self._state_list.append((i, j))

    def get_current_state(self) -> int:
        """Return the current state

        Returns:
            int
        """
        return self._current_state

    def set_current_state(self, state) -> None:
        """Set the current state for grading purpose

        Args:
            state
        """
        self._current_state = state

    def get_step_count(self) -> int:
        """Return the step count

        Returns:
            int
        """
        return self._step_count

    def get_action_space(self) -> int:
        """Return the action space

        Returns:
            int
        """
        return 4

    def get_grid_space(self) -> int:
        """Return the state space

        Returns:
            int
        """
        return len(self._state_list)

    def get_state_space(self) -> int:
        """Return the state space

        Returns:
            int
        """
        return len(self._state_list) * 2

    ##########################
    # State checker function #
    ##########################

    def _is_valid_state(self, state_coord: tuple) -> bool:
        """Check if the state is valid (within the maze and not a wall)
        Door state is not valid state.

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        if self._is_door_state(state_coord):
            return False

        if state_coord[0] < 0 or state_coord[0] >= self._maze.shape[0]:
            return False
        if state_coord[1] < 0 or state_coord[1] >= self._maze.shape[1]:
            return False
        if self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["WALL"]:
            return False
        return True

    def _is_goal_state(self, state_coord: tuple) -> bool:
        """Check if the state is a goal state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["GOAL"]

    def _is_trap_state(self, state_coord: tuple) -> bool:
        """Check if the state is a trap state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["TRAP"]

    def _is_lava_state(self, state_coord: tuple) -> bool:
        """Check if the state is a lava state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["LAVA"]

    def _is_door_state(self, state_coord: tuple) -> bool:
        """Check if the state is a door state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["DOOR"]

    def _is_key_state(self, state_coord: tuple) -> bool:
        """Check if the state is a key state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["KEY"]

    def _is_exit_state(self, state_coord: tuple) -> bool:
        """Check if the state is a exit state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["EXIT"]

    def _is_bait_state(self, state_coord: tuple) -> bool:
        """Check if the state is a bait state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["BAIT"]

    def _is_portal_state(self, state_coord: tuple) -> bool:
        """Check if the state is a portal state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["PORTAL"]

    ############################
    # Hidden Environment state #
    ############################

    @property
    def _is_closed(self):
        if self._door_state is None:
            return True
        return self._maze[self._state_list[self._door_state][0], self._state_list[self._door_state][1]] == self.OBJECT_TO_INDEX["DOOR"]

    @property
    def _is_opened(self):
        if self._door_state is None:
            return False
        return self._maze[self._state_list[self._door_state][0], self._state_list[self._door_state][1]] == self.OBJECT_TO_INDEX["EMPTY"]

    @property
    def _is_baited(self):
        if self._bait_state is None:
            return False
        return self._maze[self._state_list[self._bait_state][0], self._state_list[self._bait_state][1]] == self.OBJECT_TO_INDEX["EMPTY"]

    ########################
    # Environment function #
    ########################

    def close_door(self):
        if self._door_state is None:
            return
        self._maze[self._state_list[self._door_state][0],
                   self._state_list[self._door_state][1]] = self.OBJECT_TO_INDEX["DOOR"]
        self.render_maze()

    def open_door(self):
        if self._door_state is None:
            return
        self._maze[self._state_list[self._door_state][0],
                   self._state_list[self._door_state][1]] = self.OBJECT_TO_INDEX["EMPTY"]
        self.render_maze()

    def bite(self):
        if self._bait_state is None:
            return
        self.step_reward = self._step_reward + self._bait_step_penalty
        self._maze[self._state_list[self._bait_state][0],
                   self._state_list[self._bait_state][1]] = self.OBJECT_TO_INDEX["EMPTY"]
        self.render_maze()

    def place_bait(self):
        if self._bait_state is None:
            return
        self.step_reward = self._step_reward
        self._maze[self._state_list[self._bait_state][0],
                   self._state_list[self._bait_state][1]] = self.OBJECT_TO_INDEX["BAIT"]
        self.render_maze()

    def _get_next_state(self, state_coord: tuple, action: int) -> tuple:
        """Get the next state given the current state and action
        If the next hit the wall and the current state is portal state,
        then return the coordinate of the other portal

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
        if not self._is_valid_state(next_state_coord) and self._is_portal_state(state_coord):
            next_state_coord = self.portal_next_state[state_coord]
        if not self._is_valid_state(next_state_coord):
            next_state_coord = state_coord
        return tuple(next_state_coord)

    def step(self, action: int) -> tuple:
        """Take a step in the environment
        Refer to GridWorld in homework 1 and homework 2 implement the step function using the helper function

        Args:
            action (int)

        Returns:
            tuple: next_state, reward, done, truncation
        """
        # TODO implement the step function here
        raise NotImplementedError

    def reset(self) -> int:
        """Reset the environment

        Returns:
            int: initial state
        """
        # TODO implement the reset function here
        raise NotImplementedError

    #############################
    # Visualize the environment #
    #############################

    def __str__(self):
        """Return the maze as a string"""
        maze_str = f"Size: {self._maze.shape}\n"
        current_state_position = self._state_list[self._current_state]
        for i in range(self._maze.shape[0]):
            for j in range(self._maze.shape[1]):
                if (i, j) == current_state_position:
                    maze_str += "A"
                else:
                    maze_str += self.OBJECT_INDEX_TO_CHAR[self._maze[i, j]]
            maze_str += "\n"
        return maze_str

    def render_maze(self):
        num_colors = len(self.OBJECT_INDEX_TO_CHAR) - 1
        grid_colors = COLORS[:num_colors]
        cmap = colors.ListedColormap(grid_colors)
        self.ax.imshow(self._maze, cmap=cmap, vmin=0, vmax=num_colors)

    def render_init(self, title="GridWorld"):
        plt.close("all")

        self.fig, self.ax = plt.subplots(
            figsize=(self._maze.shape[1], self._maze.shape[0]))
        self.render_maze()
        self.ax.grid(which="major", axis="both",
                     linestyle="-", color="gray", linewidth=2)
        self.ax.set_xticks(np.arange(-0.5, self._maze.shape[1], 1))
        self.ax.set_yticks(np.arange(-0.5, self._maze.shape[0], 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(length=0)
        self.state_to_text = {}
        self.previous_state = None
        text_count = 0

        for i in range(self._maze.shape[0]):
            for j in range(self._maze.shape[1]):
                if self._maze[i, j] == 1:
                    continue

                state = self._state_list.index((i, j))
                label = f"{state}"

                self.state_to_text[state] = text_count
                text_count += 1

                self.ax.text(
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

    def visualize(self, filename=None):
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def set_text_color(self, state, color):
        text_id = self.state_to_text[state]
        text = "Agent" if color == "b" else str(state)
        self.ax.texts[text_id].set(c=color, text=text)

    def rgb_render(
        self,
    ) -> np.ndarray | None:
        """Render the environment as RGB image

        Args:
            title (str | None, optional): Title. Defaults to None.
        """
        if self.previous_state is not None:
            self.set_text_color(self.previous_state, "k")
        self.set_text_color(self._current_state, "b")
        self.previous_state = self._current_state

        if self._step_count == 0:
            plt.pause(1)
        else:
            plt.pause(0.25)

    def get_rgb(self) -> np.ndarray:
        if self.previous_state is not None:
            self.set_text_color(self.previous_state, "k")
        self.set_text_color(self._current_state, "b")
        self.previous_state = self._current_state
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        data = np.asarray(buf)
        return data


class GridWorldEnv(gym.Env):
    def __init__(self, maze_file, goal_reward, trap_reward, step_reward, exit_reward, bait_reward, bait_step_penalty, max_step, render_mode="human") -> None:
        super(GridWorldEnv, self).__init__()
        self.render_mode = render_mode
        # Initialize the GridWorld
        self.grid_world = GridWorld(maze_file, goal_reward, trap_reward,
                                    step_reward, exit_reward, bait_reward, bait_step_penalty, max_step)

        self.metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 60}
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.grid_world.get_action_space())
        self.observation_space = spaces.Discrete(self.grid_world.get_state_space())

    def reset(self, seed=None, **kwds: Any):
        # Reset the GridWorld
        # * Reset the GridWorld Environment
        # next_state = self.grid_world.reset()
        next_state = self.grid_world.reset()
        return next_state, {}

    def step(self, action):
        # Execute one step in the GridWorld
        next_state, reward, done, trucated = self.grid_world.step(action)
        return next_state, reward, done, trucated, {}

    def render(self, mode="human"):
        # print(self.render_mode)
        # Implement rendering here if needed
        if self.render_mode == "ansi":
            print(self.grid_world)
        if self.render_mode == "human":
            self.grid_world.rgb_render()

    def seed(self, seed):
        # Implement seeding here if needed
        pass

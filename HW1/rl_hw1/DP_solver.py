import numpy as np

from gridworld import GridWorld

import queue

class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        (next_state, reward, is_done) = self.grid_world.step(state, action)
        return reward + self.discount_factor * self.get_values()[next_state] * (1-is_done)


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        arr = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
        return sum(arr)/ float(len(arr))

    def run(self) -> None:
        while True:
          delta = 0.0
          copy = np.zeros(self.grid_world.get_state_space())
          for state in range(self.grid_world.get_state_space()):
            old_v = self.get_values()[state]
            new_v = self.get_state_value(state)
            copy[state] = new_v
            delta = max(delta, abs(old_v - new_v))

          self.values = copy
          if delta < self.threshold:
            break


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        return self.get_q_value(state, self.policy[state])

    def policy_evaluation(self):
        while True:
          delta = 0.0
          copy = np.zeros(self.grid_world.get_state_space())

          for state in range(self.grid_world.get_state_space()):
            old_v = self.get_values()[state]
            new_v = self.get_state_value(state)
            copy[state] = new_v
            delta = max(delta, abs(old_v - new_v))

          self.values = copy
          if delta < self.threshold:
            break

    def policy_improvement(self) -> bool:
        policy_stable = True
        copy = np.zeros(self.grid_world.get_state_space())

        for state in range(self.grid_world.get_state_space()):
          old_action = self.get_policy()[state]
          q_values = np.array([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
          new_action = np.argmax(q_values)
          copy[state] = new_action
          if new_action != old_action:
            policy_stable = False

        self.policy = copy
        return policy_stable

    def run(self) -> None:
        while True:
          self.policy_evaluation()
          result = self.policy_improvement()
          if result == True:
            break


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        arr = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
        return max(arr)

    def policy_evaluation(self):
        while True:
          delta = 0.0
          copy = np.zeros(self.grid_world.get_state_space())

          for state in range(self.grid_world.get_state_space()):
            old_v = self.get_values()[state]
            new_v = self.get_state_value(state)
            copy[state] = new_v
            delta = max(delta, abs(old_v - new_v))

          self.values = copy
          if delta < self.threshold:
            break

    def policy_improvement(self):
        copy = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
          q_values = np.array([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
          copy[state] = np.argmax(q_values)
        self.policy = copy

    def run(self) -> None:
        self.policy_evaluation()
        self.policy_improvement()

class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        arr = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
        return max(arr)

    def get_state_value_and_next_state(self, state: int) -> (float, int, bool):
        arr = np.array([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
        max_a = np.argmax(arr)
        (next_state, _, is_done) = self.grid_world.step(state, max_a)

        return (np.max(arr), next_state, is_done)

    def policy_improvement(self):
        copy = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
          q_values = np.array([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
          copy[state] = np.argmax(q_values)
        self.policy = copy

    
    def run(self) -> None:
      #Method 2: Prioritized sweeping
      heap = queue.PriorityQueue()

      for state in range(self.grid_world.get_state_space()):
        old_v = self.get_values()[state]
        new_v = self.get_state_value(state)
        self.values[state] = new_v
        heap.put((abs(old_v - new_v)*-1, state))

      while True:
        _, state = heap.get()
        old_v = self.get_values()[state]
        new_v = self.get_state_value(state)
        self.values[state] = new_v
        heap.put((abs(old_v - new_v)*-1, state))

        smallest = heap.get()
        heap.put(smallest)
        if smallest[0]*-1 < self.threshold:
          break

      self.policy_improvement()
      
      '''
      #Method 1: In-place DP
      while True:
        delta = 0.0

        for state in range(self.grid_world.get_state_space()):
          old_v = self.get_values()[state]
          new_v = self.get_state_value(state)
          self.values[state] = new_v
          delta = max(delta, abs(old_v - new_v))

        if delta < self.threshold:
          break

      self.policy_improvement()

      #Method 2: Prioritized sweeping
      heap = queue.PriorityQueue()

      for state in range(self.grid_world.get_state_space()):
        old_v = self.get_values()[state]
        new_v = self.get_state_value(state)
        self.values[state] = new_v
        heap.put((abs(old_v - new_v)*-1, state))

      while True:
        _, state = heap.get()
        old_v = self.get_values()[state]
        new_v = self.get_state_value(state)
        self.values[state] = new_v
        heap.put((abs(old_v - new_v)*-1, state))

        smallest = heap.get()
        heap.put(smallest)
        if smallest[0]*-1 < self.threshold:
          break

      self.policy_improvement()

      #Method 3: self-defined
      condition = [False for i in range(self.grid_world.get_state_space())]
      while sum(condition) < self.grid_world.get_state_space():
          for state in range(self.grid_world.get_state_space()):
            if condition[state] == True:
              continue
            old_v = self.get_values()[state]
            new_v = self.get_state_value(state)
            self.values[state] = new_v
            condition[state] = (abs(old_v - new_v) < self.threshold)

      self.policy_improvement()

      #Method 4: real-time DP
      condition = [False for i in range(self.grid_world.get_state_space())]
      while sum(condition) < self.grid_world.get_state_space():
        for state in range(self.grid_world.get_state_space()):

          is_done = False
          while not is_done:
            old_v = self.values[state]
            new_v, next_s ,next_d = self.get_state_value_and_next_state(state)
            self.values[state] = new_v
            condition[state] = (abs(old_v - new_v) < self.threshold)

            # move to next
            state = next_s
            is_done = next_d
      self.policy_improvement()
      '''

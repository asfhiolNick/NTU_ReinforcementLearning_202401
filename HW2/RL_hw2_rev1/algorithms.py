import numpy as np
import json
from collections import deque

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """


    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        current_state = self.grid_world.get_current_state()  # Get the current state

        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]
        action = self.rng.choice(self.action_space, p=action_probs)

        next_state, reward, done = self.grid_world.step(action)
        if done:
            self.episode_counter +=1
        return next_state, reward, done


class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        current_state = self.grid_world.reset()

        def is_encountered(arr, idx):
            for i in range(0,idx):
              if arr[i][0] == arr[idx][0]:
                return True
            return False

        Returns = [[] for _ in range(self.grid_world.get_state_space())]
        while self.episode_counter < self.max_episode:
            hist = [(self.grid_world.get_current_state(), -100)]
            while True:
              next_state, reward, done = self.collect_data()
              hist.append((next_state, reward))
              if done == True:
                break

            G = 0
            for t in range(len(hist)-2, -1, -1):
              G = self.discount_factor * G + hist[t+1][1]
              if is_encountered(hist,t) == False:
                S_t = hist[t][0]
                Returns[S_t].append(G)
                self.values[S_t] = sum(Returns[S_t])/len(Returns[S_t])


class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        current_state = self.grid_world.reset()

        while self.episode_counter < self.max_episode:
            S = self.grid_world.get_current_state()
            while True:
              next_state, reward, done = self.collect_data()
              self.values[S] = self.values[S] + self.lr * (reward + (1-done) * self.discount_factor * self.values[next_state] - self.values[S])
              S = next_state

              if done == True:
                break

class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        current_state = self.grid_world.reset()

        while self.episode_counter < self.max_episode:
            hist = [(self.grid_world.get_current_state(), -100)]
            S = self.grid_world.get_current_state()
            T = 2200000000
            for t in range(0, 2200000000):
              if t < T:
                next_state, reward, done = self.collect_data()
                hist.append((next_state, reward))
                if done == True:
                  T = t+1

              tau = t-self.n+1
              if tau >= 0:
                G = 0
                for i in range(tau+1, min(T,tau+self.n)+1):
                  G = G + pow(self.discount_factor, i-tau-1) * hist[i][1]

                if tau+self.n < T:
                  G = G + pow(self.discount_factor, self.n) * self.values[hist[tau+self.n][0]]

                self.values[hist[tau][0]] = self.values[hist[tau][0]] + self.lr * (G-self.values[hist[tau][0]])

              if tau == T-1:
                break


# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0, seed: int = 1):
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy
        self.rng = np.random.default_rng(seed)

    def get_policy_index(self) -> np.ndarray:
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index

    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values


class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        G = 0
        for t in range(len(action_trace)-1, -1, -1):
          S_t, A_t, R_t = state_trace[t], action_trace[t], reward_trace[t+1]
          G = self.discount_factor * G + R_t
          self.q_values[S_t, A_t] = self.q_values[S_t, A_t] + self.lr* (G - self.q_values[S_t, A_t])

    def policy_improvement(self) -> None:
        policy_ans  = self.get_policy_index()
        self.policy = np.ones((self.state_space, self.action_space)) * (self.epsilon / self.action_space)
        self.policy[np.arange(self.state_space), policy_ans] += (1 - self.epsilon)

    def run(self, max_episode=1000) -> None:
        iter_episode = 0
        current_state = self.grid_world.reset()
        while iter_episode < max_episode:
            state_trace  = [self.grid_world.get_current_state()]
            action_trace = []
            reward_trace = [0] #index padding
            while True:
              action_probs = self.policy[self.grid_world.get_current_state()]
              action = self.rng.choice(self.action_space, p=action_probs)
              action_trace.append(action)
              next_state, reward, done = self.grid_world.step(action)
              
              reward_trace.append(reward)
              state_trace.append(next_state)
              if done == True:
                iter_episode +=1
                break

            self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()


class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def run(self, max_episode=1000) -> None:
        iter_episode = 0
        current_state = self.grid_world.reset()

        while iter_episode < max_episode:
            state = self.grid_world.get_current_state()

            action_probs = np.ones(self.action_space) * (self.epsilon / self.action_space)
            action_probs[self.q_values[state].argmax()] += (1 - self.epsilon)
            action = self.rng.choice(self.action_space, p=action_probs)

            while True:
              next_state, reward, done = self.grid_world.step(action)

              next_action_probs = np.ones(self.action_space) * (self.epsilon / self.action_space)
              next_action_probs[self.q_values[next_state].argmax()] += (1 - self.epsilon)
              next_action = self.rng.choice(self.action_space, p=next_action_probs)

              self.q_values[state, action] = self.q_values[state, action] + self.lr*(reward + (1-done)*self.discount_factor*self.q_values[next_state, next_action] - self.q_values[state, action])

              state = next_state
              action = next_action
              if done == True:
                iter_episode += 1
                break


class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        raise NotImplementedError

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        raise NotImplementedError

    def policy_eval_improve(self, s, a, r, ns, d) -> None:
        self.q_values[s][a] = self.q_values[s][a] + self.lr * ( r+ (1-d) *self.discount_factor *self.q_values[ns].max() - self.q_values[s][a] )

    def run(self, max_episode=1000) -> None:
        iter_episode = 0
        current_state = self.grid_world.reset()
        transition_count = 0
        while iter_episode < max_episode:
          state = self.grid_world.get_current_state()

          while True:
            action_probs = np.ones(self.action_space) * (self.epsilon / self.action_space)
            action_probs[self.q_values[state].argmax()] += (1 - self.epsilon)
            action = self.rng.choice(self.action_space, p=action_probs)

            next_state, reward, done = self.grid_world.step(action)
            self.buffer.append((state, action, reward, next_state, done))
            transition_count += 1

            if transition_count % self.update_frequency == 0:
              for _ in range(self.sample_batch_size):
                rand_idx = self.rng.choice(len(self.buffer))
                (s,a,r,ns,d) = self.buffer[rand_idx]
                self.policy_eval_improve(s,a,r,ns,d)

            state = next_state
            if done == True:
              iter_episode += 1
              break

import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore


class ContinuousActionEnv(object):
    """
    对于连续动作环境的封装
    Wrapper for continuous action environment.
    """

    def __init__(self):
        self.env = EnvCore()
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agent):
            # physical action space
            u_action_space = spaces.Dict(
                    {
                        "p_thresholds": spaces.Box(
                            low=0, high=1, shape=(5,), dtype=float
                        ),
                        "n_thresholds": spaces.Box(
                            low=0, high=1, shape=(5,), dtype=float
                        ),
                        "matrix": spaces.Box(
                            low=0, high=1, shape=(5,), dtype=float
                        ),
                        "propose": spaces.Discrete(2),
                    }
                )

            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Dict({
                    'ranking_difference': spaces.Box(low=-10, high=10, shape=(self.env.n_member, 5, 2), dtype=float),
                    'thresholds': spaces.Dict({
                        'p_thresholds': spaces.Box(low=0, high=1, shape=(5,), dtype=float),
                        'n_thresholds': spaces.Box(low=0, high=1, shape=(5,), dtype=float)
                    })
                })
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码

        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of environment, there are 2 agents inside, and each agent's action is a 5-dimensional one_hot encoding
        """
        results = self.env.step(actions)
        obs, rews, dones, infos = results
        dones = [dones for _ in range(5)]
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass

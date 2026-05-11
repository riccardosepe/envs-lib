from functools import lru_cache

import numpy as np
from gymnasium import Env, spaces

from libs.envs.envs.common import BaseEnv, EnvStepException


# from ..common import BaseEnv


class BlockStackEnv(Env, BaseEnv):
    """
    A simple block stacking environment. There are four blocks: 'a', 'b', 'c', 'd'. The goal is to stack them in a
    specific order. The agent can pick up a block and place it on top of another block or on the table.
    However, after a block is placed on top of another block, it cannot be moved again.
    The episode ends when all blocks are stacked in the correct order or after a maximum number of steps.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_blocks=4, goal_configuration=None, *args, **kwargs):
        super().__init__()
        assert num_blocks >= 2, "There must be at least 2 blocks to stack."
        self.num_blocks = num_blocks
        self.blocks = [chr(ord('a') + i) for i in range(num_blocks)]
        self._blocks_indices = {b: i for i, b in enumerate(self.blocks)}
        self._support = None
        self.done = False
        self.t = None
        self.state_space = spaces.Discrete(self._oeis_a000262(num_blocks))
        self.action_space = spaces.Discrete(num_blocks ** 2)
        self._max_episode_length = 4 * num_blocks
        self._action_map = None
        self._inverse_action_map = None
        self._build_action_map()
        self._la = None
        self._block_value = 1 / num_blocks
        self._goal_key = goal_configuration
        if self._goal_key is None:
            self._goal_key = "".join(self.blocks)  # e.g. "abcd" for num_blocks=4
        assert len(self._goal_key) == self.num_blocks
        self.goal_configuration = self.from_key(self._goal_key)

    @staticmethod
    @lru_cache(maxsize=None)
    def _oeis_a000262(n):
        """
        Number of sets of lists from Online Encyclopedia of Integer Sequences (OEIS) A000262.
        """
        if n <= 1:
            return 1
        return (2 * n - 1) * BlockStackEnv._oeis_a000262(n - 1) \
            - (n - 1) * (n - 2) * BlockStackEnv._oeis_a000262(n - 2)

    def _build_action_map(self):
        if self._action_map is not None:
            raise RuntimeError("Action map already built")
        self._action_map = dict()
        self._inverse_action_map = dict()
        idx = 0
        for source_idx in range(self.num_blocks):
            for dest_idx in range(self.num_blocks):
                if source_idx != dest_idx:
                    self._action_map[idx] = (source_idx, dest_idx)
                    self._inverse_action_map[(source_idx, dest_idx)] = idx
                    idx += 1

            self._action_map[idx] = (source_idx, self.num_blocks)
            self._inverse_action_map[(source_idx, self.num_blocks)] = idx
            idx += 1

    def reset(self, *args, **kwargs):
        # Initial state: all blocks on the table
        self._support = self.num_blocks * np.ones(self.num_blocks, dtype=int)
        self.done = False
        self._la = None
        self.t = 0
        return self.observation(), {}

    def observation(self):
        return self._support.copy()

    def step(self, action):
        if action not in self.legal_actions:
            raise EnvStepException

        source_idx, dest_idx = self._action_map[action]
        self._support[source_idx] = dest_idx

        self.t += 1

        if np.array_equal(self._support, self.goal_configuration):
            self.done = True

        truncated = False
        if self.t >= self._max_episode_length:
            truncated = True

        self._la = action
        return self.observation(), self.reward(), self.done, truncated, {}

    def render(self):
        print(self.to_key(self._support))

    def block_is_free(self, block_idx):
        if block_idx == self.num_blocks:
            return True
        return not np.any(self._support == block_idx)

    @property
    def legal_actions(self):
        legal = []
        for i, (source_idx, dest_idx) in self._action_map.items():
            if not self.block_is_free(source_idx) or dest_idx == self._support[source_idx]:
                continue
            else:
                if self.block_is_free(dest_idx) or dest_idx == self.num_blocks:
                    legal.append(i)

        return legal

    @property
    def _last_action(self):
        return self._la

    @property
    def adversarial(self):
        return False

    def backup(self):
        backup = BaseEnv.backup(self)
        backup.update({
            'state': self._support.copy(),
        })
        return backup

    def load(self, checkpoint):
        try:
            self._support = checkpoint['state'].copy()
            self.done = checkpoint['done']
            self._la = checkpoint['last_action']
            self.t = checkpoint['t']
        except KeyError as e:
            print(e)
            return False
        return True

    def game_result(self, **kwargs):
        pass

    def reward(self, **kwargs):
        if self.done:
            return 1.0
        else:
            return -0.01

    @property
    def state_space_cardinality(self):
        return self.state_space.n

    @property
    def action_space_cardinality(self):
        return self._action_map.n

    @property
    def max_episode_length(self):
        return self._max_episode_length

    def get_action_id(self, source, dest):
        # OBS: there is a 1:1 mapping between keys and values
        if source == dest:
            return -1
        source_idx = ord(source) - ord('a')
        if dest == 'table':
            dest_idx = self.num_blocks
        else:
            dest_idx = ord(dest) - ord('a')
        return self._inverse_action_map[(source_idx, dest_idx)]

    @staticmethod
    def to_key(support):
        num_blocks = len(support)
        blocks = [chr(i + ord('a')) for i in range(len(support))]
        free_blocks_idxs = list(set(range(num_blocks)) - set(support))
        stacks = []

        for fb in free_blocks_idxs:
            stack = [fb]
            x = fb
            while True:
                y = support[x]
                stack.append(y)

                if y == num_blocks:
                    break
                x = y
            stacks.append(''.join(blocks[i] for i in reversed(stack[:-1])))
        return '|'.join(sorted(stacks))

    @staticmethod
    def from_key(key):
        # convert back to dictionary
        stacks = key.split('|')
        num_blocks = sum([len(s) for s in stacks])
        support = np.zeros(num_blocks, dtype=int)
        for stack in stacks:
            for i, block in enumerate(stack):
                block_idx = ord(block) - ord('a')
                if i == 0:
                    support[block_idx] = num_blocks
                else:
                    support[block_idx] = ord(stack[i - 1]) - ord('a')
        return support

    @staticmethod
    def encode(support):
        n = len(support)

        x = np.zeros((n, n + 1), dtype=np.float32)
        x[np.arange(n), support] = 1.0
        return x.flatten()

    @staticmethod
    def decode(inp):
        support = np.where(inp == 1)[1]
        assert len(support) == inp.shape[0]
        return support


if __name__ == '__main__':
    env = BlockStackEnv(num_blocks=4)
    obs, _ = env.reset()
    print("Initial state:", obs)
    env.render()

    done = False
    while not done:
        src = input('Enter source block: ')
        dst = input('Enter destination block: ')

        act_id = env.get_action_id(src, dst)
        while act_id not in env.legal_actions:
            print('Invalid action')
            src = input('Enter source block: ')
            dst = input('Enter destination block: ')

            act_id = env.get_action_id(src, dst)

        obs, reward, done, truncated, _ = env.step(act_id)
        env.render()
        print(f"Reward: {reward:.3f}")

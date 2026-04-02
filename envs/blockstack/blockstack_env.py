import math

import numpy as np
import scipy
from gymnasium import Env, spaces

from libs.envs.envs.common import BaseEnv


# from ..common import BaseEnv


class BlockStackEnv(Env, BaseEnv):
    """
    A simple block stacking environment. There are four blocks: 'a', 'b', 'c', 'd'. The goal is to stack them in a
    specific order. The agent can pick up a block and place it on top of another block or on the table.
    However, after a block is placed on top of another block, it cannot be moved again.
    The episode ends when all blocks are stacked in the correct order or after a maximum number of steps.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_blocks=4, *args, **kwargs):
        super().__init__()
        assert num_blocks >= 2, "There must be at least 2 blocks to stack."
        self.num_blocks = num_blocks
        self.blocks = [chr(ord('a')+i) for i in range(num_blocks)]
        self.state = None
        self.done = False
        self.t = None
        self.tallest_stack = None
        self.state_space = spaces.Discrete(self._bell_number(num_blocks))
        self.action_space = spaces.Discrete(num_blocks**2)
        self._max_episode_length = 4*num_blocks
        self._action_map = None
        self._inverse_action_map = None
        self._build_action_map()
        self._la = None
        self._block_value = 1 / num_blocks

    @staticmethod
    def _bell_number(n):
        if n == 0:
            return 1
        else:
            return sum(BlockStackEnv._bell_number(k) * math.comb(n-1, k) for k in range(n))

    def _build_action_map(self):
        if self._action_map is not None:
            raise RuntimeError("Action map already built")
        self._action_map = dict()
        self._inverse_action_map = dict()
        idx = 0
        for source in self.blocks:
            for dest in self.blocks:
                if source != dest:
                    self._action_map[idx] = (source, dest)
                    self._inverse_action_map[(source, dest)] = idx
                    idx += 1

            self._action_map[idx] = (source, 'table')
            self._inverse_action_map[(source, 'table')] = idx
            idx += 1


    def reset(self, *args, **kwargs):
        # Initial state: all blocks on the table
        self.state = {block: 'table' for block in self.blocks}
        self.done = False
        self._la = None
        self.t = 0
        self.tallest_stack = 1
        return self.observation(), {}

    def observation(self):
        return self.state.copy()

    def step(self, action):
        if action not in self.legal_actions:
            raise ValueError

        source, dest = self._action_map[action]
        self.state[source] = dest

        self.tallest_stack = max(map(len, self.stacks))

        self.t += 1

        if len(self.stacks) == 1:
            self.done = True

        truncated = False
        if self.t >= self._max_episode_length:
            truncated = True

        self._la = action
        return self.observation(), self.reward(), self.done, truncated, {}

    def render(self):
        print(self.stacks)

    @property
    def stacks(self):
        # Find which blocks are on the table — these are stack bases
        bases = [b for b in self.blocks if self.state[b] == 'table']

        # Build a reverse map: which block is ON me?
        on_me = {}  # block -> the block sitting on it
        for b in self.blocks:
            if self.state[b] != 'table':
                on_me[self.state[b]] = b

        # Walk up from each base
        stacks = []
        for base in bases:
            stack = [base]
            current = base
            while current in on_me:
                current = on_me[current]
                stack.append(current)
            stacks.append(stack)

        return stacks

    def block_is_free(self, block):
        if block != 'table' and block not in self.blocks:
            raise ValueError
        for b, on in self.state.items():
            if on == block:
                return False
        return True

    @property
    def legal_actions(self):
        legal = []
        for i, (source, dest) in self._action_map.items():
            if not self.block_is_free(source) or dest == self.state[source]:
                continue
            else:
                if self.block_is_free(dest) or dest == 'table':
                    legal.append(i)

        return legal

    @property
    def _last_action(self):
        return self._la

    @property
    def adversarial(self):
        return False

    def backup(self):
        pass

    def load(self, checkpoint):
        pass

    def game_result(self, **kwargs):
        pass

    def reward(self, **kwargs):
        return self._block_value * self.tallest_stack

    @property
    def state_space_cardinality(self):
        return self.state_space.n

    @property
    def action_space_cardinality(self):
        return self._action_map.n

    @property
    def max_episode_length(self):
        return self._max_episode_length

    def get_action_id(self, src, dest):
        # OBS: there is a 1:1 mapping between keys and values
        return self._inverse_action_map[(src, dest)]



if __name__ == '__main__':
    env = BlockStackEnv(num_blocks=4)
    obs, _ = env.reset()
    print("Initial state:", obs)
    env.render()

    done = False
    while not done:
        src = input('Enter source block: ')
        dest = input('Enter destination block: ')

        act_id = env.get_action_id(src, dest)
        while act_id not in env.legal_actions:
            print('Invalid action')
            src = input('Enter source block: ')
            dest = input('Enter destination block: ')

            act_id = env.get_action_id(src, dest)

        obs, reward, done, truncated, _ = env.step(act_id)
        env.render()
        print(f"Reward: {reward:.3f}")

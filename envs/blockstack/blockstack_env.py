from collections import deque
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
        self.blocks = [chr(ord('a')+i) for i in range(num_blocks)]
        self.state = None
        self.done = False
        self.t = None
        self.state_space = spaces.Discrete(self._oeis_a000262(num_blocks))
        self.action_space = spaces.Discrete(num_blocks**2)
        self._max_episode_length = 4*num_blocks
        self._action_map = None
        self._inverse_action_map = None
        self._build_action_map()
        self._la = None
        self._block_value = 1 / num_blocks
        self.goal_configuration = goal_configuration
        if self.goal_configuration is None:
            self.goal_configuration = "".join(self.blocks)  # e.g. "abcd" for num_blocks=4
        assert len(self.goal_configuration) == self.num_blocks

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
        return self.observation(), {}

    def observation(self):
        return self.state.copy()

    def step(self, action):
        if action not in self.legal_actions:
            raise EnvStepException

        source, dest = self._action_map[action]
        self.state[source] = dest

        self.t += 1

        state_string = self.to_key(self.state)
        if state_string == self.goal_configuration:
            self.done = True

        truncated = False
        if self.t >= self._max_episode_length:
            truncated = True

        self._la = action
        return self.observation(), self.reward(), self.done, truncated, {}

    def render(self):
        print(self.to_key(self.state))

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
        backup = BaseEnv.backup(self)
        backup.update({
            'state': self.state.copy(),
        })
        return backup

    def load(self, checkpoint):
        try:
            self.state = checkpoint['state'].copy()
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

    def get_action_id(self, src, dest):
        # OBS: there is a 1:1 mapping between keys and values
        return self._inverse_action_map[(src, dest)]

    @staticmethod
    def to_key(state):
        # stacks is list[list[str]] from env.stacks (bottom-to-top)
        stacks = []
        state_queue = deque(state.items())
        while state_queue:
            block, location = state_queue.popleft()

            if location == 'table':
                stacks.append(block)
            else:
                found = False
                for stack in stacks:
                    if stack.endswith(location):
                        stacks.remove(stack)
                        stack = stack + block
                        stacks.append(stack)
                        found = True
                if not found:
                    state_queue.append((block, location))

        return '|'.join(sorted(''.join(s) for s in stacks))

    @staticmethod
    def from_key(key):
        # convert back to dictionary
        stacks = key.split('|')
        state = {}
        for stack in stacks:
            for i, block in enumerate(stack):
                if i == 0:
                    state[block] = 'table'
                else:
                    state[block] = stack[i-1]
        return state

    @staticmethod
    def state_to_input(state):
        blocks = list(sorted(state.keys()))
        n = len(blocks)
        idx = {b: i for i, b in enumerate(blocks)}
        idx['table'] = n

        x = np.zeros((n, n + 1), dtype=np.float32)
        for b in blocks:
            x[idx[b], idx[state[b]]] = 1.0
        return x.flatten()



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

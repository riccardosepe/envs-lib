from gymnasium import Env, spaces

from envs.common import BaseEnv


class BlockStackEnv(Env, BaseEnv):
    """
    A simple block stacking environment. There are four blocks: 'a', 'b', 'c', 'd'. The goal is to stack them in a
    specific order. The agent can pick up a block and place it on top of another block or on the table.
    However, after a block is placed on top of another block, it cannot be moved again.
    The episode ends when all blocks are stacked in the correct order or after a maximum number of steps.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.blocks = ['a', 'b', 'c']
        self.positions = ['table', 'a', 'b', 'c']
        self.state = None
        self.done = False
        self.t = None
        self.state_space = spaces.Discrete(None)
        self.action_space = spaces.Discrete(None)

    def reset(self, *args, **kwargs):
        # Initial state: all blocks on the table
        self.state = {block: 'table' for block in self.blocks}
        self.done = False
        self.t = 0
        return self.observation(), {}

    def observation(self):
        return self.state.copy()

    def step(self, action):
        pass

    def render(self):
        pass

    @property
    def legal_actions(self):
        pass

    @property
    def _last_action(self):
        pass

    @property
    def adversarial(self):
        pass

    def backup(self):
        pass

    def load(self, checkpoint):
        pass

    def game_result(self, **kwargs):
        pass

    def reward(self, **kwargs):
        pass

    @property
    def state_space_cardinality(self):
        pass

    @property
    def action_space_cardinality(self):
        pass

    @property
    def max_episode_length(self):
        return self._max_episode_length



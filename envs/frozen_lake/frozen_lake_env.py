import inspect
from copy import deepcopy
from os import path

import numpy as np
from gymnasium.envs.toy_text import FrozenLakeEnv as GymFrozenLakeEnv
from gymnasium.error import DependencyNotInstalled

from envs.common import BaseEnv

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    'simple': [
        "SFFF",
        "HFFF",
        "HFFF",
        "GFFF"
    ],
    'corridors': [
        "HFSFH",
        "HFFFH",
        "HFFFH",
        "HFFFH",
        "HFFFH",
        "HGGGH",
    ]
}

WALLS = {
    'corridors': {
        ((0, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((0, 3), (0, 4)),
        ((1, 3), (1, 4)),
        ((2, 1), (2, 2)),
        ((2, 2), (2, 3)),
        ((3, 1), (3, 2)),
        ((3, 2), (3, 3)),
        ((4, 1), (4, 2)),
        ((4, 2), (4, 3)),
        ((5, 1), (5, 2)),
        ((5, 2), (5, 3))
    }
}

class FrozenLakeEnv(BaseEnv, GymFrozenLakeEnv):
    def __init__(self, *args, p=1/3, wall_transitions=None, **kwargs):
        self._max_episode_length = kwargs.pop('max_episode_length', 1000)
        if 'map_name' in kwargs and kwargs['map_name'] is not None and kwargs['map_name'] in MAPS:
            desc = MAPS[kwargs['map_name']]
            wall_transitions = WALLS.get(kwargs['map_name'], None)
            kwargs['map_name'] = None
            kwargs['desc'] = desc
        super().__init__(*args, **kwargs)
        self._last_reward = None
        self.done = False
        self.lastaction = None
        self.is_slippery = kwargs.get('is_slippery', False)
        self.t = 0

        # wall transitions
        self.wall_transitions = wall_transitions

        ps = [(1-p)/2, p, (1-p)/2]

        self.nA = 4
        self.nS = self.nrow * self.ncol

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        def to_s(row, col):
            return row * self.ncol + col

        def inc(row, col, a):
            new_row, new_col = row, col
            if a == LEFT:
                new_col = max(col - 1, 0)
            elif a == DOWN:
                new_row = min(row + 1, self.nrow - 1)
            elif a == RIGHT:
                new_col = min(col + 1, self.ncol - 1)
            elif a == UP:
                new_row = max(row - 1, 0)

            if self.wall_transitions is not None:
                # block a movement through the wall
                if ((row, col), (new_row, new_col)) in self.wall_transitions or ((new_row, new_col), (row, col)) in self.wall_transitions:
                    new_row, new_col = row, col

            return new_row, new_col

        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = self.desc[new_row, new_col]
            terminated = bytes(new_letter) in b"GH"
            reward = float(new_letter == b"G")
            return new_state, reward, terminated

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if self.is_slippery:
                            for k, b in enumerate([(a - 1) % 4, a, (a + 1) % 4]):
                                li.append(
                                    (ps[k], *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))


    def reset(self, *args, **kwargs):
        self._last_reward = None
        self.done = False
        self.t = 0
        return super().reset(*args, **kwargs)

    @property
    def legal_actions_old(self):
        # NB:
        # - 0: Move left
        # - 1: Move down
        # - 2: Move right
        # - 3: Move up
        i, j = self.s // self.ncol, self.s % self.ncol
        actions = []
        if i > 0:
            actions.append(3)
        if i < self.nrow - 1:
            actions.append(1)
        if j > 0:
            actions.append(0)
        if j < self.ncol - 1:
            actions.append(2)
        return actions

    @property
    def legal_actions(self):
        return list(range(4))

    @property
    def _last_action(self):
        return self.lastaction

    def step(self, a):
        render_mode = self.render_mode
        self.render_mode = ''
        s, r, d, t, i = super().step(a)
        self.render_mode = render_mode
        self._last_reward = r
        self.done = d
        self.t += 1
        truncated = False
        if self.t >= self.max_episode_length:
            self.done = True
            truncated = True
        return s, r, d, truncated, i

    @property
    def adversarial(self):
        return False

    def backup(self):
        checkpoint = {
            'state': deepcopy(self.s),
            'last_action': self._last_action,
            'done': self.done,
            'reward': self.reward(),
            'player': 'Agent',
            't': self.t
        }
        return checkpoint

    def load(self, checkpoint):
        self.s = checkpoint['state']
        self.lastaction = checkpoint['last_action']
        self._last_reward = checkpoint['reward']
        self.done = checkpoint['done']
        self.t = checkpoint['t']

    def game_result(self, code=False):
        if not self.done:
            if code:
                return self.STILL_RUNNING
            else:
                return "Game still running"
        else:
            if self.desc.flatten()[self.s] == b'G':
                if code:
                    return self.WON
                else:
                    return f"You made it after {self.t} steps!"
            else:
                if code:
                    return self.LOST
                else:
                    return f"You fell into an ice pit after {self.t} steps :("

    def reward(self):
        return self._last_reward

    def next_states(self, action):
        """
        This method computes the support of the stochastic transition function of the environment, based on the action
        """
        # TODO: check if works also in other cases
        return np.unique(list(map(lambda sp: sp[1], self.P[self.s][action]))).tolist()


    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(inspect.getfile(GymFrozenLakeEnv)), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(inspect.getfile(GymFrozenLakeEnv)), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(inspect.getfile(GymFrozenLakeEnv)), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(inspect.getfile(GymFrozenLakeEnv)), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(inspect.getfile(GymFrozenLakeEnv)), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(inspect.getfile(GymFrozenLakeEnv)), "img/elf_left.png"),
                path.join(path.dirname(inspect.getfile(GymFrozenLakeEnv)), "img/elf_down.png"),
                path.join(path.dirname(inspect.getfile(GymFrozenLakeEnv)), "img/elf_right.png"),
                path.join(path.dirname(inspect.getfile(GymFrozenLakeEnv)), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        if self.wall_transitions is not None:
            wall_thickness = 4
            for y in range(self.nrow):
                for x in range(self.ncol):
                    # Draw walls (thick black lines between cells)
                    if ((y, x), (y, x + 1)) in self.wall_transitions:  # Vertical walls
                        pygame.draw.line(
                            self.window_surface, (0, 0, 0),
                            (x * self.cell_size[0] + self.cell_size[0], y * self.cell_size[1]),
                            (x * self.cell_size[0] + self.cell_size[0], (y + 1) * self.cell_size[1]),
                            wall_thickness  # Line thickness
                        )
                    if ((y, x), (y + 1, x)) in self.wall_transitions:  # Horizontal walls
                        pygame.draw.line(
                            self.window_surface, (0, 0, 0),
                            (x * self.cell_size[0], y * self.cell_size[1] + self.cell_size[1]),
                            ((x + 1) * self.cell_size[0], y * self.cell_size[1] + self.cell_size[1]),
                            wall_thickness  # Line thickness
                        )

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @property
    def max_episode_length(self):
        return self._max_episode_length


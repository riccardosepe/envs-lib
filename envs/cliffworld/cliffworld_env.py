import sys
from copy import deepcopy
from os import path

import gymnasium
import numpy as np
import pygame
from gymnasium import Env
from gymnasium.envs.registration import EnvSpec
from gymnasium.error import DependencyNotInstalled

from ..common import BaseEnv

MAPS = {""
    "base": [
        "FFFFH",
        "FFFFM",
        "FCXCT",
        "SFFFF"
    ]
}

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

key_action_map = {
    pygame.K_LEFT: 0,   # left
    pygame.K_DOWN: 1,   # down
    pygame.K_RIGHT: 2,  # right
    pygame.K_UP: 3      # up
}


def categorical_sample(prob_n, np_random: np.random.Generator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return int(np.argmax(csprob_n > np_random.random()))



class CliffWorldEnv(BaseEnv, Env):

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    BAR_HEIGHT = 40  # height of the status bar in pixels

    def __init__(
            self,
            render_mode,
            *args,
            desc=None,
            map_name="base",
            max_episode_length=None,
            **kwargs
    ):
        """
        For the moment, let's keep it simple. The possible fruit is only apples (`A` on the map) and bananas (`B`).
        """
        if max_episode_length is None:
            max_episode_length = np.inf
        self._max_episode_length = max_episode_length
        if desc is None and map_name is None:
            raise NotImplementedError
        elif desc is None:
            desc = MAPS[map_name]

        # super().__init__(*args, **kwargs)

        self.spec = EnvSpec(id="BarterGameEnv-v0")

        self.desc = desc = np.asarray(desc, dtype="c")

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.nrow, self.ncol = desc.shape

        cliffs_pos = np.argwhere(self.desc == b'C').tolist() + np.argwhere(self.desc == b'X').tolist()
        monsters_pos = np.argwhere(self.desc == b'M').tolist()[0]
        treasure_good_pos = np.argwhere(self.desc == b'T').tolist()[0]
        treasure_bad_pos = np.argwhere(self.desc == b'H').tolist()[0]
        gold_pos = np.argwhere(self.desc == b'X').tolist()[0]

        self.cliffs = {tuple(a): False for a in cliffs_pos}


        self.action_space_size = 4
        self.state_space_size = self.nrow * self.ncol

        self._la = None
        self.s = None
        self.t = 0
        self.done = False

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * self.ncol, 1024), min(64 * self.nrow, 1024))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_size = self.window_size[0], self.window_size[1] + self.BAR_HEIGHT
        self.window_surface = None
        self.clock = None

        self.apple_img = None
        self.banana_img = None
        self.coconut_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
        self.grass_img = None

    @property
    def action_space_cardinality(self):
        return self.action_space_size

    @property
    def state_space_cardinality(self):
        return self.state_space_size

    @property
    def legal_actions(self):
        return list(range(self.action_space_size))

    @property
    def _last_action(self):
        return self._la

    @property
    def adversarial(self):
        return False

    @property
    def max_episode_length(self):
        return self._max_episode_length

    def reset(self, *, seed=None, **kwargs):
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self._la = None
        self.t = 0
        self.cliffs = {k: False for k in self.cliffs.keys()}
        self.done = False
        return self.s, None

    def step(self, action):
        i, j = self.s // self.ncol, self.s % self.ncol
        ii, jj = self.inc(i, j, action)

        done = True if self.desc[ii, jj] in [b'T', b'H', b'M', b'C', b'X'] else False

        self._la = action
        self.s = ii * self.ncol + jj

        self.t += 1
        truncated = True if self.t >= self.max_episode_length else False
        self.done = done or truncated

        return self.s, self.reward(), self.done, truncated, {}

    def backup(self):
        state = {
            'state': self.s,
            'done': self.done,
            'cliffs': deepcopy(self.cliffs),
            'last_action': self._la,
            't': self.t,
            'reward': self.reward(),
            'current_player': 0,  # Not used in this environment
            'player': 'Agent',
        }
        return state

    def load(self, checkpoint):
        try:
            self.s = checkpoint['state']
            self.done = checkpoint['done']
            self.cliffs = checkpoint['cliffs']
            self._la = checkpoint['last_action']
            self.t = checkpoint['t']
        except KeyError as e:
            print(e, file=sys.stderr)
            return False
        return True

    def game_result(self, **kwargs):
        pass

    def reward_components(self):
        i, j = self.s // self.ncol, self.s % self.ncol
        rew = np.zeros((4,))
        if self.desc[i, j] == b'T':
            rew[3] = 15
        if self.desc[i, j] == b'H':
            rew[3] = 1
        if self.desc[i, j] == b'M':
            rew[2] = -2
        if self.desc[i, j] == b'C':
            rew[0] = -10
        if self.desc[i, j] == b'X':
            rew[1] = 10
            rew[0] = -10
        denom = 10 + 15
        offset = denom / 10
        rew = (rew + offset) / denom
        return rew

    def reward(self, ):
        return sum(self.reward_components())

    def inc(self, i, j, a):
        if a == LEFT:
            j = max(j-1, 0)
        if a == DOWN:
            i = min(i+1, self.nrow-1)
        if a == RIGHT:
            j = min(j+1, self.ncol-1)
        if a == UP:
            i = max(i-1, 0)

        return i, j

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            raise NotImplementedError
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

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
                pygame.display.set_caption("Fruit Collection")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        bar_rect = pygame.Rect(0, 0, self.window_size[0], self.BAR_HEIGHT)
        pygame.draw.rect(self.window_surface, (50, 50, 50), bar_rect)  # Dark background
        font = pygame.font.SysFont("Arial", 40, bold=True)
        # Render text (replace self.score/self.steps with your real variables)

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.grass_img is None:
            file_name = path.join(path.dirname(__file__), "../img/grass.png")
            self.grass_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "../img/warehouse.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "../img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "../img/elf_left.png"),
                path.join(path.dirname(__file__), "../img/elf_down.png"),
                path.join(path.dirname(__file__), "../img/elf_right.png"),
                path.join(path.dirname(__file__), "../img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]
        if self.apple_img is None:
            file_name = path.join(path.dirname(__file__), "../img/apple.png")
            self.apple_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.banana_img is None:
            file_name = path.join(path.dirname(__file__), "../img/banana.png")
            self.banana_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.coconut_img is None:
            file_name = path.join(path.dirname(__file__), "../img/coconut.png")
            self.coconut_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1] + self.BAR_HEIGHT)

                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.grass_img, pos)
                if self.desc[y, x] == b'H':
                    text_surface = font.render('t', True, (0, 0, 0))
                    self.window_surface.blit(text_surface, pos)
                elif self.desc[y, x] == b'T':
                    text_surface = font.render('T', True, (0, 0, 0))
                    self.window_surface.blit(text_surface, pos)
                elif self.desc[y, x] == b'X':
                    text_surface = font.render('X', True, (0, 0, 0))
                    self.window_surface.blit(text_surface, pos)
                elif self.desc[y, x] == b'M':
                    text_surface = font.render('M', True, (0, 0, 0))
                    self.window_surface.blit(text_surface, pos)
                elif self.desc[y, x] == b'C':
                    text_surface = font.render('C', True, (0, 0, 0))
                    self.window_surface.blit(text_surface, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1] + self.BAR_HEIGHT)
        last_action = self._la if self._la is not None else 1
        elf_img = self.elf_images[last_action]

        # if desc[bot_row][bot_col] == b"A":
        #     self.window_surface.blit(self.cracked_hole_img, cell_rect)
        # else:
        #     self.window_surface.blit(elf_img, cell_rect)

        self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )


if __name__ == "__main__":
    env = CliffWorldEnv(#map_name="8x15",
                        render_mode='human',
                        max_episode_length=100,
                        )
    env.reset()
    env.render()
    done = False
    while not done:
        action = None
        while action is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN and event.key in key_action_map:
                    action = key_action_map[event.key]

        # action = int(input("Insert action: "))
        t = _, r, done, _, _ = env.step(action)
        print(t)
        env.render()
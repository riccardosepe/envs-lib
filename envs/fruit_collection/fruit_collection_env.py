from os import path

import gymnasium
import numpy as np
import pygame
from gymnasium import Env
from gymnasium.envs.registration import EnvSpec
from gymnasium.error import DependencyNotInstalled

from ..common.base_env import BaseEnv

MAPS = {
    "base": [
        "SFFBF",
        "FAFAF",
        "FAFFF",
        "FFBFG"
    ],
    "8x15": [
        "FFFFBFFFFFFBFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFAFFFFAFFFFF",
        "FFFFFFFFFFFAFGF",
        "FFFFFAFFFFFFFFF",
        "FSFFFFFFFFAFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFBFFFFFF"
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



class FruitCollectionEnv(BaseEnv, Env):

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

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
        For the moment let's keep it simple. The possible fruit is only apples (`A` on the map) and bananas (`B`).
        """
        if max_episode_length is None:
            max_episode_length = np.inf
        self._max_episode_length = max_episode_length
        if desc is None and map_name is None:
            raise NotImplementedError
        elif desc is None:
            desc = MAPS[map_name]

        super().__init__(*args, **kwargs)

        self.spec = EnvSpec(id="FruitCollection-v0")

        self.desc = desc = np.asarray(desc, dtype="c")

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.nrow, self.ncol = desc.shape

        apples_pos = np.argwhere(self.desc == b'A').tolist()
        bananas_pos = np.argwhere(self.desc == b'B').tolist()

        self.apples = {tuple(a): False for a in apples_pos}
        self.bananas = {tuple(b): False for b in bananas_pos}

        self.apple_weight = 1
        self.apple_val = 2
        self.bananas_weight = 2
        self.banana_val = 5

        self.knapsack_max_capacity = 5
        self.knapsack_weight = 0
        self.knapsack_value = 0
        self._max_theoretical_value = 12  # 2 * 5 + 1 * 2

        self.action_space_size = 4
        self.state_space_size = self.nrow * self.ncol

        self.last_action = None
        self.s = None
        self._last_pickup = 0
        self.t = 0

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * self.ncol, 1024), min(64 * self.nrow, 1024))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None

        self.apple_img = None
        self.banana_img = None
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
        return self.last_action

    @property
    def adversarial(self):
        return False

    @property
    def max_episode_length(self):
        return self._max_episode_length

    def reset(self, *, seed=None, **kwargs):
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self._last_pickup = 0
        self.last_action = None
        self.t = 0
        self.knapsack_weight = 0
        self.knapsack_value = 0
        self.apples = {k: False for k in self.apples.keys()}
        self.bananas = {k: False for k in self.bananas.keys()}
        return self.s, None

    def _pickup(self, i, j):
        if (i, j) in self.apples and not self.apples[(i, j)]:
            self.apples[(i, j)] = True
            self._last_pickup = self.apple_val

        if (i, j) in self.bananas and not self.bananas[(i, j)]:
            self.bananas[(i, j)] = True
            self._last_pickup = self.banana_val

    def step(self, action):
        i, j = self.s // self.ncol, self.s % self.ncol
        ii, jj = self.inc(i, j, action)
        self._pickup(ii, jj)

        done = True if self.desc[(ii, jj)] == b"G" else False

        self.last_action = action
        self.s = ii * self.ncol + jj

        self.t += 1
        truncated = True if self.t >= self.max_episode_length else False
        done = done or truncated

        return self.s, self.reward(), done, truncated, {}

    def backup(self):
        pass

    def load(self, checkpoint):
        pass

    def game_result(self, **kwargs):
        pass

    def reward(self):
        ii, jj = self.s // self.ncol, self.s % self.ncol

        components = [sum(self.apples.values()) * self.apple_val,]

        if self.desc[(ii, jj)] == b"G":
            return self.knapsack_value / self._max_theoretical_value

        else:
            return 0.

    def reward_dense(self):
        a, b, f = 0, 0, 0
        ii, jj = self.s // self.ncol, self.s % self.ncol

        if (ii, jj) in self.apples and not self.apples[(ii, jj)]:
            self.apples[(ii, jj)] = True
            a = self.apple_val

        if (ii, jj) in self.bananas and not self.bananas[(ii, jj)]:
            self.bananas[(ii, jj)] = True
            b = self.banana_val

        if self.desc[(ii, jj)] == b"G":
            tot_a = len(self.apples)
            tot_b = len(self.bananas)
            coll_a = sum(self.apples.values())
            coll_b = sum(self.bananas.values())
            a_val = self.apple_val
            b_val = self.banana_val
            f = (a_val + b_val) * (coll_a * a_val + coll_b * b_val) / (tot_a * a_val + tot_b * b_val)

        # NB: either a, b or f is different from zero
        return sum((a, b, f))

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

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.grass_img is None:
            file_name = path.join(path.dirname(__file__), "../img/grass.png")
            self.grass_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "../img/goal.png")
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

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.grass_img, pos)
                if desc[y][x] == b"A" and not self.apples[(y, x)]:
                    self.window_surface.blit(self.apple_img, pos)
                elif desc[y][x] == b"B" and not self.bananas[(y, x)]:
                    self.window_surface.blit(self.banana_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.last_action if self.last_action is not None else 1
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
    env = FruitCollectionEnv(map_name="8x15",
                             render_mode='human',
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
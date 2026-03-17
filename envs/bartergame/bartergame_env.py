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

MAPS = {
    "base": [
        "SFFBF",
        "FAFAF",
        "FAFFF",
        "FFBFG"
    ],
    "base_c": [
        "SFFBF",
        "FAFAF",
        "FAFFC",
        "FFBFG"
    ],
    "4x5": [
        "SAFBF",
        "FFFBF",
        "FAFFF",
        "FAFFG"
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
    ],
    "map_small": [
        "SFFBF",
        "FAFAF",
        "FAFFF",
        "FFBFG"
    ],
    "map_small_c": [
        "SFFBF",
        "FAFAF",
        "FAFFC",
        "FFBFG"
    ],
    "map_medium": [
        "SFFBAFFA",
        "FFFFFFFF",
        "AFAFFGFF",
        "FFABFFFB",
        "FFFFFFFF",
        "FFFFFFFF"
    ],
    "map_medium_c": [
        "SFFBAFCA",
        "FFFFFFFF",
        "AFAFFGFF",
        "FFABFFFB",
        "FFFFFFFF",
        "FFFCFFFF"
    ],
    "map_large": [
        "SFFFFFFFFFFAFFF",
        "FFFFFBFFFFFFFFF",
        "FFFFFFFFFFFFFBF",
        "AFFFFFFFFFFFFFA",
        "AFFFFFFAFFFBFFF",
        "FFFFFFFFFFFFFAF",
        "FFFFFFFFFFFFFFF",
        "BFFFFFFBFFAFFFG"
    ],
    "map_large_c": [
        "SFFFFFFFFFFAFFC",
        "FFFFFBFFFFFFFFF",
        "FFFFFCFFFFFFFBF",
        "AFFFFFFFFFFFFFA",
        "AFFFFFFAFFFBFFF",
        "FFFFFFFFFFCFFAF",
        "FFFFFFFFFFFFFFF",
        "BFFFFFFBFFAFFFG"
    ],
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



class BarterGameEnv(BaseEnv, Env):
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

        apples_pos = np.argwhere(self.desc == b'A').tolist()
        bananas_pos = np.argwhere(self.desc == b'B').tolist()
        coconuts_pos = np.argwhere(self.desc == b'C').tolist()

        self.apples = {tuple(a): False for a in apples_pos}
        self.bananas = {tuple(b): False for b in bananas_pos}
        self.coconuts = {tuple(c): False for c in coconuts_pos}

        self._dropped_apples = 0
        self._dropped_bananas = 0

        self.action_space_size = 4
        self.state_space_size = self.nrow * self.ncol

        self._knapsack = {
            'apple': 0,
            'banana': 0,
            'coconut': 0
        }
        self._warehouse = {
            'apple': 0,
            'banana': 0,
            'coconut': 0
        }
        # Let's just specify the ratios of the fruit values, then normalize
        banana_vs_apple = 3
        coconut_vs_banana = 4

        if len(self.coconuts) > 0:
            coconut_vs_apple = banana_vs_apple * coconut_vs_banana
            denom = 1 + banana_vs_apple + coconut_vs_apple
            self._apple_coeff = 1 / denom
            self._banana_coeff = banana_vs_apple / denom
            self._coconut_coeff = coconut_vs_apple / denom
        else:
            denom = 1 + banana_vs_apple
            self._apple_coeff = 1 / denom
            self._banana_coeff = banana_vs_apple / denom
            self._coconut_coeff = 0  # No coconuts, so no coefficient

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
    def max_episode_length(self):
        return self._max_episode_length

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
    def board_is_clean(self):
        """
        The board is clean if all apples and all bananas have been picked up and unloaded.
        """
        return (all(self.apples.values())
                and all(self.bananas.values())
                and all(self.coconuts.values())
                and sum(self._knapsack.values()) == 0)

    @property
    def game_locked(self):
        """
        The game cannot proceed further if all apples have been picked up and unloaded.
        """
        return all(self.apples.values()) and self._knapsack['apple'] == 0

    def reset(self, *, seed=None, **kwargs):
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self._la = None
        self.t = 0
        self.apples = {k: False for k in self.apples.keys()}
        self.bananas = {k: False for k in self.bananas.keys()}
        self.coconuts = {k: False for k in self.coconuts.keys()}
        self.done = False
        self._dropped_apples = 0
        self._dropped_bananas = 0
        self._knapsack = {
            'apple': 0,
            'banana': 0,
            'coconut': 0
        }
        self._warehouse = {
            'apple': 0,
            'banana': 0,
            'coconut': 0
        }
        return self.s, None

    def _drop_apple(self, i, j, n=1):
        old_pos = None
        for k, v in self.apples.items():
            if v:
                self._knapsack['apple'] -= n
                self._dropped_apples += n
                old_pos = k
                break
        # self.apples.pop(old_pos)
        # self.apples[(i, j)] = False

    def _drop_banana(self, i, j, n=1):
        old_pos = None
        for k, v in self.bananas.items():
            if v:
                self._knapsack['banana'] -= n
                self._dropped_bananas += n
                old_pos = k
                break

    def _pickup(self, i, j):
        if (i, j) in self.apples and not self.apples[(i, j)] and sum(self._knapsack.values()) < 2:
            self.apples[(i, j)] = True
            self._knapsack['apple'] += 1

        if (i, j) in self.bananas and not self.bananas[(i, j)] and self._knapsack['apple'] > 0:
            self.bananas[(i, j)] = True
            self._knapsack['banana'] += 1
            self._drop_apple(i, j)

        if (i, j) in self.coconuts and not self.coconuts[(i, j)] and self._knapsack['banana'] > 0:
            self.coconuts[(i, j)] = True
            self._knapsack['coconut'] += 1
            self._drop_banana(i, j)

    def _unload(self, i, j):
        if self.desc[(i, j)] == b"G":
            self._warehouse['banana'] += self._knapsack['banana']
            self._warehouse['apple'] += self._knapsack['apple']
            self._warehouse['coconut'] += self._knapsack['coconut']
            self._knapsack['banana'] = 0
            self._knapsack['apple'] = 0
            self._knapsack['coconut'] = 0

    def _on_warehouse(self):
        i, j = self.s // self.ncol, self.s % self.ncol
        return self.desc[i][j] == b'G'

    def step(self, action):
        i, j = self.s // self.ncol, self.s % self.ncol
        ii, jj = self.inc(i, j, action)
        self._pickup(ii, jj)
        self._unload(ii, jj)

        done = True if self.board_is_clean else False

        self._la = action
        self.s = ii * self.ncol + jj

        self.t += 1
        truncated = True if self.t >= self.max_episode_length else False
        self.done = done or truncated

        return self.s, self.reward(done), self.done, truncated, {}

    def backup(self):
        state = {
            'state': self.s,
            'done': self.done,
            'apples': deepcopy(self.apples),
            'bananas': deepcopy(self.bananas),
            'coconuts': deepcopy(self.coconuts),
            'knapsack': deepcopy(self._knapsack),
            'warehouse': deepcopy(self._warehouse),
            'dropped_apples': self._dropped_apples,
            'dropped_bananas': self._dropped_bananas,
            # 'alpha': self._banana_coeff,
            'last_action': self._la,
            't': self.t,
            'reward': self.reward(self.done),
            'current_player': 0,  # Not used in this environment
            'player': 'Agent',
        }
        return state

    def load(self, checkpoint):
        try:
            self.s = checkpoint['state']
            self.done = checkpoint['done']
            self.apples = checkpoint['apples']
            self.bananas = checkpoint['bananas']
            self.coconuts = checkpoint['coconuts']
            self._knapsack = checkpoint['knapsack']
            self._warehouse = checkpoint['warehouse']
            # self._banana_coeff = checkpoint['alpha']
            self._dropped_apples = checkpoint['dropped_apples']
            self._dropped_bananas = checkpoint['dropped_bananas']
            # self._apple_coeff = 1 - self._banana_coeff
            self._la = checkpoint['last_action']
            self.t = checkpoint['t']
        except KeyError as e:
            print(e, file=sys.stderr)
            return False
        return True

    def game_result(self, **kwargs):
        pass

    def reward(self, done):
        if done:
            collected_apples = self._warehouse['apple'] / (len(self.apples)-self._dropped_apples)
            collected_bananas = self._warehouse['banana'] / (len(self.bananas)-self._dropped_bananas)
            if len(self.coconuts) > 0:
                collected_coconuts = self._warehouse['coconut'] / len(self.coconuts)
            else:
                collected_coconuts = 0
                assert self._dropped_bananas == 0

            # fruit reward is in [0, 1]
            fruit_reward = (collected_apples * self._apple_coeff
                            + collected_bananas * self._banana_coeff
                            + collected_coconuts * self._coconut_coeff)

            # time penalty
            # time_penalty = 0.2 if self.t > self.max_episode_length // 2 else 0
            time_penalty = 0.1 * (self.t / self.max_episode_length)

            return fruit_reward - time_penalty
        else:
            return 0

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
        font = pygame.font.SysFont("Arial", 20)
        # Render text (replace self.score/self.steps with your real variables)

        w = 'Warehouse' if self.desc.size > 20 else 'W'
        k = 'Knapsack' if self.desc.size > 20 else 'K'

        stats_text = f"{w}: A:{self._warehouse['apple']} B:{self._warehouse['banana']} | {k}: A:{self._knapsack['apple']} B:{self._knapsack['banana']}"
        text_surface = font.render(stats_text, True, (255, 255, 255))
        self.window_surface.blit(text_surface, (10, 10))

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
                if (y, x) in self.apples and not self.apples[(y, x)]:
                    self.window_surface.blit(self.apple_img, pos)
                elif (y, x) in self.bananas and not self.bananas[(y, x)]:
                    self.window_surface.blit(self.banana_img, pos)
                elif (y, x) in self.coconuts and not self.coconuts[(y, x)]:
                    self.window_surface.blit(self.coconut_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

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
    env = BarterGameEnv(#map_name="8x15",
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
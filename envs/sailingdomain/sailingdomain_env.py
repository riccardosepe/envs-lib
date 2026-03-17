"""
This file holds a custom environment for the SAIL domain, which is a gridworld-like environment used for reinforcement
learning experiments. The environment has a starting position on the left, a goal position on the right,
and some obstacles and treasures in between. Additionally, the wind is flowing from left to right, meaning that
the agent can only move right, up-right, or down-right.
An episode ends when the agent reaches the goal or after it hits an obstacle. There is no timestep limit because
the wind will eventually push the agent to the goal or to an obstacle.
The environment subclasses Env from gymnasium and BaseEnv from ..common.base_env.
The environment defines the action space, observation space, reset, step, render methods, and other necessary properties.
The reward structure is as follows:
- Reaching the goal: +1
- Collecting a treasure of type A: +0.2
- Collecting a treasure of type B: +0.1
- Hitting an obstacle: -1
- Any other move: 0
"""
from contextlib import closing
from io import StringIO
from os import path

import numpy as np
from gymnasium import spaces, Env, utils
from gymnasium.error import DependencyNotInstalled

from ..common import BaseEnv

maps = {
    "base": [
        ".BB...",
        ".@....",
        "S....G",
        "...@..",
        "..AA..",
    ],
    "5x10": [
        "...BBBB...",
        "..@..@....",
        "S....B...G",
        "...@..@...",
        "...A......",
    ],
    "map_1": [
    "..B.B.B...",
    "..........",
    "S.A......G",
    "...B......",
    "..........",
    ],
    "map_2": [
        "..BBBBBB..",
        "..@@@@@@..",
        "S..@A....G",
        "..A....B..",
        "..........",
    ],
    "map_2_1": [
        "...@A.....",
        "..BBBBBB..",
        "S.@@@@@@.G",
        "..A....B..",
        "..........",
    ],
    "map_2_2": [
        "...@A.....",
        "..A....B..",
        "S.BBBBBB.G",
        "..@@@@@@..",
        "..........",
    ],

    "map_2_3": [
        "...@A.....",
        "..A....B..",
        "S........G",
        "..BBBBBB..",
        "..@@@@@@..",
    ],
    "map_3": [
        "..BBBBBB..",
        "...@@@@@..",
        "S..@A....G",
        "..A.......",
        "..........",
    ],
}

class SailingDomainEnv(Env, BaseEnv):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 4,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()

        # Setup
        self.actions = {
            "up": -1,
            "right": 0,
            "down": 1,
        }
        self._act_id_to_action = {i: a for i, a in enumerate(self.actions.keys())}
        self._action_to_act_id = {v: k for k, v in self._act_id_to_action.items()}
        self.action_space = spaces.Discrete(len(self.actions))

        map_name = kwargs.pop("map_name", "base")
        if map_name not in maps:
            raise ValueError(f"Invalid map name: {map_name}")

        if 'randomize' in kwargs and kwargs['randomize']:
            base_name = map_name
            map_variants = [name for name in maps.keys() if name.startswith(base_name + "_")]
            if len(map_variants) > 0:
                map_name = np.random.choice(map_variants)

        # Map elements
        self._setup_map(map_name)

        self._obstacle_reward = -0.1
        self._goal_reward = 0.4
        self._non_goal_reward = 0.0
        self._a_reward = 0.3
        self._b_reward = 0.1

        self._terminate_on_obstacle = kwargs.pop("terminate_on_obstacle", False)

        # State
        self.state = None
        self._done = False
        self.t = None
        self._la = None
        self._last_reward = None

        # Rendering
        self._scale = 2
        self.render_mode = kwargs.pop("render_mode", None)
        if self.render_mode not in self.metadata["render_modes"] and self.render_mode is not None:
            raise ValueError(f"Invalid render mode: {self.render_mode}")
        self.window_surface = None
        self.window_size = (self._scale * 32 * self.cols, self._scale * 32 * self.rows)
        self.cell_size = (
            self.window_size[0] // self.cols,
            self.window_size[1] // self.rows,
        )
        self.window_surface = None
        self.clock = None
        self.whirl_img = None
        self.water_img = None
        self.boat_img = None
        self.chest_A_img = None
        self.chest_B_img = None
        self.goal_img = None
        self.start_img = None
        self.coast_img = None
        self.wind_img = None

    def _setup_map(self, map_name):
        self.desc = np.array(maps[map_name], dtype="c")
        self.rows, self.cols = self.desc.shape
        self.state_space = spaces.Discrete(self.rows * self.cols)

        i, j = np.where(self.desc == b'S')
        self.start_pos = (i.item(), j.item())
        i, j = np.where(self.desc == b'G')
        self.goal_pos = (i.item(), j.item())
        i, j = np.where(self.desc == b'@')
        self.obstacles = set(zip(i.tolist(), j.tolist()))
        i, j = np.where(self.desc == b'A')
        self.treasures_a = set(zip(i.tolist(), j.tolist()))
        i, j = np.where(self.desc == b'B')
        self.treasures_b = set(zip(i.tolist(), j.tolist()))
        treasures = self.treasures_a | self.treasures_b
        assert len(treasures) == len(self.treasures_a) + len(self.treasures_b), "Treasures overlap!"

        self._taken_treasures = dict.fromkeys(self.treasures_a | self.treasures_b, False)

    def reset(self, *args, **kwargs):
        self.state = self.start_pos
        self._done = False
        self._la = None
        self._last_reward = None
        self.t = 0
        self._taken_treasures = dict.fromkeys(self.treasures_a | self.treasures_b, False)
        return self.observation(), {}


    def observation(self, integer=False):
        if integer:
            return self.state[0] * self.cols + self.state[1]
        else:
            return self.state

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        dy = self.actions[self._act_id_to_action[action]]
        dx = 1  #  the wind always pushes right
        new_row = max(0, min(self.rows - 1, self.state[0] + dy))
        new_col = min(self.cols - 1, self.state[1] + dx)
        new_pos = (new_row, new_col)

        self.state = new_pos
        self.t += 1
        self._la = action

        self._last_reward = self.reward(new_pos)
        done = self._done = self.done(new_pos)

        return self.observation(), self._last_reward, done, False, {}

    @property
    def legal_actions(self):
        actions = self._action_to_act_id.copy()
        if self.state[0] == 0:
            actions.pop("up")
        if self.state[0] == self.rows - 1:
            actions.pop("down")
        return list(actions.values())

    @property
    def _last_action(self):
        return self._la

    @property
    def adversarial(self):
        return False

    def game_result(self, **kwargs):
        pass

    def reward(self, new_pos, **kwargs):
        reward_vector = np.zeros(self.reward_space_cardinality)
        if new_pos in self.obstacles:
            reward_vector[0] = self._obstacle_reward
        elif new_pos in self.treasures_a and not self._taken_treasures[new_pos]:
            self._taken_treasures[new_pos] = True
            reward_vector[1] = self._a_reward
        elif new_pos in self.treasures_b and not self._taken_treasures[new_pos]:
            self._taken_treasures[new_pos] = True
            reward_vector[2] = self._b_reward
        elif new_pos == self.goal_pos:
            reward_vector[3] = self._goal_reward
        elif new_pos[1] == self.cols - 1:
            reward_vector[3] = self._non_goal_reward
        else:
            pass

        return reward_vector

    def done(self, new_pos):
        if new_pos[1] == self.cols - 1 or (self._terminate_on_obstacle and new_pos in self.obstacles):
            return True
        else:
            return False

    @property
    def state_space_cardinality(self):
        return self.state_space.n

    @property
    def action_space_cardinality(self):
        return self.action_space.n

    @property
    def reward_space_cardinality(self):
        # OBS: the number of entries should be the number of different concepts
        #  for example, in a single cell there technically could be both a golden and a silver chest, or
        #  a chest and the goal, or a chest and an obstacle
        #  However, the terminal state rewards are mutually exclusive, either you get to the goal and get 1
        #  or you end up on the shore and get 0.05
        return 4

    @property
    def max_episode_length(self):
        return self._max_episode_length

    def backup(self):
        ckpt = {
            'state': self.state,
            'last_action': self._la,
            'done': self._done,
            't': self.t,
            'reward': self._last_reward,
            'taken_treasures': self._taken_treasures.copy(),
            'player': 'Agent'
        }
        return ckpt

    def load(self, checkpoint):
        try:
            skip = False
            if 'map_name' in checkpoint:
                skip = True
                self._setup_map(checkpoint['map_name'])

            self.state = checkpoint['state']
            self._la = checkpoint['last_action']
            self._last_reward = checkpoint['reward']
            self._done = checkpoint['done']
            self.t = checkpoint['t']
            if not skip:
                self._taken_treasures = checkpoint['taken_treasures'].copy()
        except KeyError as e:
            raise RuntimeError("Invalid checkpoint format") from e

    def render(self):
        if self.state is None:
            print("Environment not initialized. Call reset() first.")
            return

        board = np.array([['.' for _ in range(self.cols)] for _ in range(self.rows)], dtype="c")
        for (r, c) in self.obstacles:
            board[r, c] = '@'
        for (r, c) in self.treasures_a:
            board[r, c] = 'A'
        for (r, c) in self.treasures_b:
            board[r, c] = 'B'
        board[self.goal_pos[0], self.goal_pos[1]] = 'G'
        board[self.state[0], self.state[1]] = 'R'  # Robot position
        board[self.start_pos[0], self.start_pos[1]] = 'S'  # Start position

        if self.render_mode == 'ansi':
            # Print the board
            print(self._render_text(board))

        elif self.render_mode == 'human':
            return self._render_gui(self.desc, self.render_mode)

    def _render_text(self, board):
        desc = board.tolist()
        outfile = StringIO()

        row, col = self.state
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self._la is not None:
            outfile.write(f"  ({self._act_id_to_action[self._la]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def _render_gui(self, board, mode='human'):
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
                pygame.display.set_caption("Sail Domain")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.whirl_img is None:
            file_name = path.join(path.dirname(__file__), "img/whirl_1.png")
            self.whirl_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.water_img is None:
            self.water_img = []
            for i in range(1, 4):
                file_name = path.join(path.dirname(__file__), f"img/water_{i}.png")
                img = pygame.transform.scale(
                    pygame.image.load(file_name).convert_alpha(),
                    self.cell_size
                )

                # Soft white overlay
                overlay = pygame.Surface(self.cell_size, pygame.SRCALPHA)
                overlay.fill((255, 255, 255, 70))  # try 40–70

                # IMPORTANT: no special_flags here
                img.blit(overlay, (0, 0))

                self.water_img.append(img)
        if self.coast_img is None:
            self.coast_img = []
            file_name = path.join(path.dirname(__file__), "img/coast_1.png")
            self.coast_img.append(pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            ))
            file_name = path.join(path.dirname(__file__), "img/coast_2.png")
            self.coast_img.append(pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            ))
            file_name = path.join(path.dirname(__file__), "img/coast_3.png")
            self.coast_img.append(pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            ))
        if self.boat_img is None:
            file_name = path.join(path.dirname(__file__), "img/boat.png")
            self.boat_img = pygame.transform.scale(
                pygame.image.load(file_name), ((self.cell_size[0] // 4)*3, (self.cell_size[1] // 4)*3)
            )
        if self.chest_A_img is None:
            file_name = path.join(path.dirname(__file__), "img/chest_1.png")
            self.chest_A_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.chest_B_img is None:
            file_name = path.join(path.dirname(__file__), "img/chest_2.png")
            self.chest_B_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/start.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.wind_img is None:
            file_name = path.join(path.dirname(__file__), "img/wind.png")
            self.wind_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = board.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.rows):
            for x in range(self.cols):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.water_img[x%3], pos)
                if x == y == 0:
                    self.window_surface.blit(self.wind_img, pos)
                if desc[y][x] == b"@":
                    self.window_surface.blit(self.whirl_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)
                elif desc[y][x] == b"A" and not self._taken_treasures[(y, x)]:
                    self.window_surface.blit(self.chest_A_img, pos)
                elif desc[y][x] == b"B" and not self._taken_treasures[(y, x)]:
                    self.window_surface.blit(self.chest_B_img, pos)
                elif x == self.cols-1:
                    if desc[y][x] == b"G":
                        self.window_surface.blit(self.goal_img, pos)
                    elif y == self.rows-1:
                        self.window_surface.blit(self.coast_img[0], pos)
                    elif y == 0:
                        self.window_surface.blit(self.coast_img[2], pos)
                    else:
                        self.window_surface.blit(self.coast_img[1], pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # Plot the grid lines
        grid_color = (100, 100, 100)
        # grid_color = (0, 0, 0)
        line_width = max(1, int(self._scale/3))

        # Vertical lines
        for x in range(self.cols + 1):
            pygame.draw.line(
                self.window_surface,
                grid_color,
                (x * self.cell_size[0], 0),
                (x * self.cell_size[0], self.rows * self.cell_size[1]),
                line_width
            )

        # Horizontal lines
        for y in range(self.rows + 1):
            pygame.draw.line(
                self.window_surface,
                grid_color,
                (0, y * self.cell_size[1]),
                (self.cols * self.cell_size[0], y * self.cell_size[1]),
                line_width
            )

        # paint the boat
        boat_row, boat_col = self.state
        off_y = (self.cell_size[0] - self.boat_img.get_height())
        off_x = (self.cell_size[1] - self.boat_img.get_width()) // 2
        cell_rect = (boat_col * self.cell_size[0] + off_y, boat_row * self.cell_size[1] + off_x)
        self.window_surface.blit(self.boat_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def decode_action_human(self, a):
        return self._act_id_to_action[a]


if __name__ == '__main__':
    import pygame

    key_action_map = {
        pygame.K_UP: 0,
        pygame.K_RIGHT: 1,
        pygame.K_DOWN: 2,
    }

    env = SailingDomainEnv(
        render_mode='human',
        map_name='map_2',
    )
    env.reset()
    env.render()
    done = False

    ret = 0

    while not done:
        action = None
        while action is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN and event.key in key_action_map:
                    action = key_action_map[event.key]
        _, reward, done, _, _ = env.step(action)
        ret += reward
        env.render()
    print("Episode finished.")
    print("Return:", ret)
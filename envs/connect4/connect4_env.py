import json
from copy import deepcopy

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..common import BaseEnv
from ..common.base_env import EnvStepException
from ..common.constants import EMPTY_CELL, WHITE, BLACK


def render_latex(text, color=(0, 0, 0), font_size=22):
    fig = Figure(figsize=(.9, 0.45), dpi=140)
    fig.patch.set_alpha(0)

    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis("off")

    # convert hex color to RGB tuple if needed
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        color = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))

    # IMPORTANT: draw text in center of normalized axes coordinates
    ax.text(
        0.5,
        0.5,
        f"${text}$",
        fontsize=font_size,
        color=[c / 255 for c in color],
        ha="center",
        va="center"
    )

    return canvas


class Connect4Env(Env, BaseEnv):
    """
        GameState for the Connect 4 game.
        The board is represented as a 2D array (rows and columns).
        Each entry on the array can be:
            EMPTY_CELL = empty    (.)
            WHITE = player 1 (X)
            BLACK = player 2 (O)

        Winner can be:
             None = No winner (yet)
            -1 = Draw
             WHITE = player 1 (X)
             BLACK = player 2 (O)
    """

    ANSI_PIECES = {
        EMPTY_CELL: '.',
        WHITE: 'O',
        BLACK: 'X'
    }

    def __init__(self, width=7, height=6, connect=4, **kwargs):
        self.num_players = 2
        self._max_episode_length = kwargs.pop('max_episode_length', None)
        if self._max_episode_length is None:
            self._max_episode_length = np.inf

        self.num_columns = width
        self.column_height = height
        self.connect = connect

        self.board = None
        self.current_player = None
        self._la = None
        self._done = None
        self.t = 0
        # Set human/agent colors using constants (human was first player in original)
        self.human_color = None
        self.agent_color = None

        self.action_space = Discrete(self.num_columns)

        # GUI related defaults (initialized before first render)
        self.window_size = (min(64 * self.num_columns, 512), min(64 * self.column_height, 512))
        self._rectangles = {}  # dict of cell coordinate lists to highlight (e.g. winning 4-in-a-row)
        self._alternatives = {}  # dict of alternative actions to render
        self.cell_size = (self.window_size[0] // max(1, self.num_columns),
                          self.window_size[1] // max(1, self.column_height))
        self.index_width = 24
        self.status_bar_height = 50
        self.window_surface = None
        self.clock = None
        self.render_mode = kwargs.pop('render_mode', None)
        self._font_size = 22

    def __copy__(self):
        new_env = Connect4Env(self.num_columns, self.column_height, self.connect)
        new_env.board = deepcopy(self.board)
        new_env._done = self._done
        new_env.current_player = self.current_player
        new_env.human_color = self.human_color
        new_env.agent_color = self.agent_color
        new_env.t = self.t
        new_env._la = self._la
        return new_env

    @property
    def observation(self):
        return deepcopy(self.board), self.current_player

    @property
    def other_player(self):
        return self.opponent_color(self.current_player)

    @property
    def legal_actions(self):
        """
        :returns: array with all possible moves, index of columns which aren't full
        """
        # column is available if top cell (height-1) is EMPTY_CELL
        return [col for col in range(self.num_columns) if self.board[col][self.column_height - 1] == EMPTY_CELL]

    @property
    def _last_action(self):
        return self._la

    @property
    def adversarial(self):
        return True

    @property
    def state_space_cardinality(self):
        raise NotImplementedError

    @property
    def action_space_cardinality(self):
        return self.num_columns

    @property
    def max_episode_length(self):
        return self._max_episode_length

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reward(self):
        if not self.done:
            return 0
        else:
            if self.done == self.human_color:
                return -1
            elif self.done == self.agent_color:
                return 1
            else:
                return 0

    def _player_label(self):
        return 'Human' if self.current_player == self.human_color else 'Agent'

    def backup(self):
        state = super().backup()
        state.update({
            'state': self.observation,
            'board': deepcopy(self.board),
            'current_player': self.current_player,
        })
        return state

    def load(self, checkpoint):
        try:
            super().load(checkpoint)
            board, player = checkpoint['state']
            self.board = deepcopy(board)
            self.current_player = checkpoint['current_player']
            return True
        except KeyError:
            return False

    def game_result(self, human_readable=False):
        if human_readable:
            if not self.done:
                return "Game still running..."
            if self.done == self.human_color:
                return "You won!"
            if self.done == self.agent_color:
                return "You lost!"
            return "Draw."
        else:
            if not self.done:
                return 0
            return 1 if self.done == self.human_color else 0

    # ------------------------------------------------------------------
    # Game logic
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        assert 'agent_color' in kwargs
        agent_color = kwargs['agent_color']

        self.agent_color = agent_color
        self.human_color = self.opponent_color(agent_color)

        # Always start from a clean board; callers that need to resume
        # from a checkpoint call env.load(ckpt_dict) afterwards, following
        # the same pattern as BreakthroughEnv (and the BaseEnv contract).
        self.board = np.full((self.num_columns, self.column_height), EMPTY_CELL)
        self.current_player = WHITE
        self._done = 0
        self.t = 0
        self._la = None
        return self.observation

    @classmethod
    def build_checkpoint(cls, checkpoint_id: int):
        """
        Load a Connect4 saved-game state by its integer ID.

        Parameters
        ----------
        checkpoint_id : int

        Returns
        -------
        ckpt_dict : dict
        agent_color : int  (WHITE or BLACK)
        """
        ckpt_id = str(checkpoint_id)
        with open(f'config/connect4_scenarios.json') as f:
            data = json.load(f)

        if data[ckpt_id]['current_player'] == 'white':
            current_player = WHITE
        elif data[ckpt_id]['current_player'] == 'black':
            current_player = BLACK
        else:
            raise ValueError("Unknown player color in checkpoint")

        if data[ckpt_id]['agent_color'] == 'white':
            agent_color = WHITE
        elif data[ckpt_id]['agent_color'] == 'black':
            agent_color = BLACK
        else:
            raise ValueError("Unknown agent color in checkpoint")
        ckpt = {
            "current_player": current_player,
            'done': False,
            'last_action': data[ckpt_id].get('last_action', None),
            't': data[ckpt_id].get('t', 0),
        }

        # Parse board from scenario (array of column strings)
        board_data = data[ckpt_id]['board']
        height = len(board_data[0]) if board_data else 6
        width = len(board_data)
        board = np.full((width, height), EMPTY_CELL, dtype=int)

        for col_idx, col_str in enumerate(board_data):
            for row_idx, char in enumerate(col_str):
                if char.lower() == 'w':
                    board[col_idx][row_idx] = WHITE
                elif char.lower() == 'b':
                    board[col_idx][row_idx] = BLACK

        ckpt['board'] = board
        state = (board, current_player)
        ckpt['state'] = state

        return ckpt, agent_color

    def step(self, action):
        """
        Drop a chip in the given column.
        Compatible with BreakthroughEnv's step() signature.
        Returns: (observation, reward, done, truncated, info)
        """
        if self.done:
            raise RuntimeError("Connect4Env was terminated")

        if action not in self.legal_actions:
            raise EnvStepException(
                f"Invalid move. Tried column {action}. "
                f"Legal actions are: {self.legal_actions}"
            )

        (index,) = next((i for i, v in np.ndenumerate(self.board[action]) if v == 0))

        # Place the chip
        self.board[action][index] = self.current_player
        self._la = action

        # Check game termination
        if self.four_in_a_row(self.board, self.current_player):
            self._done = self.current_player
        elif all(self.board[col][self.column_height - 1] != EMPTY_CELL for col in range(self.num_columns)):
            self._done = True

        self.current_player = WHITE if self.current_player == BLACK else BLACK

        # Increase turn counter
        self.t += 1

        # No truncation logic for Connect 4
        truncated = False
        info = {}

        return self.observation, self.reward(), self.done, truncated, info

    def four_in_a_row(self, board, player):
        """Checks if `player` has a 4-piece line"""
        return (
                any(
                    all(board[c, r] == player)
                    for c in range(self.num_columns)
                    for r in (list(range(n, n + 4)) for n in range(self.column_height - 4 + 1))
                )
                or any(
            all(board[c, r] == player)
            for r in range(self.column_height)
            for c in (list(range(n, n + 4)) for n in range(self.num_columns - 4 + 1))
        )
                or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + 4), range(co, co + 4))
                for ro in range(0, self.num_columns - 4 + 1)
                for co in range(0, self.column_height - 4 + 1)
            )
        )
                or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + 4), range(co + 4 - 1, co - 1, -1))
                for ro in range(0, self.num_columns - 4 + 1)
                for co in range(0, self.column_height - 4 + 1)
            )
        )
        )

    def is_on_board(self, x, y):
        return 0 <= x < self.num_columns and 0 <= y < self.column_height

    def decode_action_human(self, action, board=None):
        if board is None:
            board = self.board
        row = next((i for i, v in np.ndenumerate(board[action]) if v == 0))[0] + 1

        if row > self.column_height:
            raise ValueError("Column is full")

        col_char = chr(ord('A') + action)
        return f"{col_char}{row}"

    def decode_action_input(self, action) -> str:
        """Column letter only — the landing row is implied by board state."""
        return chr(ord('A') + action)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, **kwargs):
        """
        Support 'ansi' textual rendering and 'human' pygame GUI rendering.
        """
        self._rectangles = {}
        self._alternatives = {}
        if self.render_mode == 'ansi':
            # Print top row first to match console representation
            for x in range(self.column_height - 1, -1, -1):
                row_str = "".join(self.ANSI_PIECES[self.board[col][x]] for col in range(self.num_columns))
                print(row_str)
            print(''.join(chr(ord('A') + i) for i in range(self.num_columns)))
        elif self.render_mode == 'human':
            if 'rectangles' in kwargs:
                self._rectangles = kwargs['rectangles']
            if 'alternatives' in kwargs:
                # alternatives are just the action (i.e., the column)
                act_col = kwargs['alternatives']['action']
                foil_col = kwargs['alternatives']['foil']

                act_row = next((i for i, v in np.ndenumerate(self.board[act_col]) if v == 0))[0]
                foil_row = next((i for i, v in np.ndenumerate(self.board[foil_col]) if v == 0))[0]

                if act_row >= self.column_height or foil_row >= self.column_height:
                    raise ValueError("Alternative action column is full")

                self._alternatives = {'action': (act_col, act_row), 'foil': (foil_col, foil_row)}
            return self._render_gui()
        else:
            raise NotImplementedError(f"Render self.render_mode '{self.render_mode}' not supported.")

    def _render_gui(self):
        try:
            import pygame
            def latex_to_surface(latex_str, color=(0, 0, 0), font_size=20):
                canvas = render_latex(latex_str, color=color, font_size=font_size)
                # Draw the canvas
                canvas.draw()

                # Get RGBA buffer from canvas
                renderer = canvas.get_renderer()
                raw_data = renderer.buffer_rgba()

                size = canvas.get_width_height()

                # Convert to pygame surface
                surface = pygame.image.frombuffer(raw_data, size, "RGBA").convert_alpha()

                return surface

        except ImportError as e:
            raise ImportError("pygame is not installed. Run `pip install pygame`.") from e

        # Initialize window if needed
        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Connect4")
            total_width = self.window_size[0] + self.index_width
            total_height = self.window_size[1] + self.index_width + self.status_bar_height
            self.window_surface = pygame.display.set_mode((total_width, total_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Colors
        BG_COLOR = (50, 50, 50)
        CELL_LIGHT = "#156082"
        CELL_DARK = "#156082"
        EMPTY_COLOR = "#FFFFFF"
        WHITE_COLOR = "#FFC002"
        BLACK_COLOR = "#C00000"

        self.window_surface.fill(BG_COLOR)

        # Draw grid and discs. We map board[col][row], and draw rows top->bottom
        for row in range(self.column_height):
            for col in range(self.num_columns):
                # compute cell rectangle
                x = self.index_width + col * self.cell_size[0]
                y = self.status_bar_height + row * self.cell_size[1]
                rect = pygame.Rect(x, y, self.cell_size[0], self.cell_size[1])
                color = CELL_LIGHT if (row + col) % 2 == 0 else CELL_DARK
                pygame.draw.rect(self.window_surface, color, rect)

                # piece at visual row: top row corresponds to board row index column_height-1
                board_row_idx = self.column_height - 1 - row
                piece = self.board[col][board_row_idx]
                if piece != EMPTY_CELL:
                    # draw disc centered in the cell
                    center = (x + self.cell_size[0] // 2, y + self.cell_size[1] // 2)
                    radius = min(self.cell_size) // 2 - 4
                    piece_color = WHITE_COLOR if piece == WHITE else BLACK_COLOR
                    pygame.draw.circle(self.window_surface, piece_color, center, radius)
                    # add border
                    # pygame.draw.circle(self.window_surface, (0, 0, 0), center, radius, 2)
                else:
                    if (col, board_row_idx) in self._alternatives.values():

                        center = (
                            x + self.cell_size[0] // 2,
                            y + self.cell_size[1] // 2
                        )
                        radius = min(self.cell_size) // 2 - 4

                        # Draw special disc
                        pygame.draw.circle(self.window_surface, (255, 255, 255), center, radius)
                        color = BLACK_COLOR if self.current_player == BLACK else WHITE_COLOR
                        pygame.draw.circle(self.window_surface, color, center, radius, 5)

                        # Determine which latex expression to render
                        if self._alternatives["action"] == (col, board_row_idx):
                            latex_code = r"a^*"
                            latex_color = (0, 0, 0)

                        elif self._alternatives["foil"] == (col, board_row_idx):
                            latex_code = r"\bar{a}"
                            latex_color = (0, 0, 0)

                        else:
                            raise RuntimeError

                        # Render latex
                        latex_surface = latex_to_surface(latex_code, color=color, font_size=22)

                        # Center it inside the disc
                        latex_rect = latex_surface.get_rect(center=center)

                        self.window_surface.blit(latex_surface, latex_rect)
                    else:
                        # draw an empty slot marker
                        center = (x + self.cell_size[0] // 2, y + self.cell_size[1] // 2)
                        radius = min(self.cell_size) // 2 - 6
                        pygame.draw.circle(self.window_surface, EMPTY_COLOR, center, radius)

        # Column labels
        try:
            col_font = pygame.font.SysFont("arial", self._font_size, bold=True)
            for col in range(self.num_columns):
                label = chr(ord('A') + col)
                text_surface = col_font.render(label, True, (255, 255, 255))
                x = self.index_width + col * self.cell_size[0] + self.cell_size[0] // 2 - text_surface.get_width() // 2
                y = (self.status_bar_height +
                     self.column_height * self.cell_size[1] +
                     (self.index_width // 2 - text_surface.get_height() // 2))
                self.window_surface.blit(text_surface, (x, y))
        except Exception:
            # Fonts may fail on some systems; ignore if so
            col_font = None

        # Row labels (numbers starting from 1 at the bottom), rendered in the left index area.
        try:
            # if font creation failed above, try to create one here
            if 'col_font' not in locals() or col_font is None:
                col_font = pygame.font.SysFont("arial", self._font_size, bold=True)
            for row in range(self.column_height):
                # visual row loop: row=0 is top; we want label numbers starting from 1 at bottom
                label_num = self.column_height - row  # bottom -> 1
                label = str(label_num)
                text_surface = col_font.render(label, True, (255, 255, 255))
                # center the row label vertically within the cell and horizontally within index_width
                x = (self.index_width // 2) - (text_surface.get_width() // 2)
                y = (self.status_bar_height +
                     row * self.cell_size[1] +
                     (self.cell_size[1] // 2 - text_surface.get_height() // 2))
                self.window_surface.blit(text_surface, (x, y))
        except Exception:
            # ignore font/rendering errors for row labels
            pass

        # Status bar
        try:
            status_font = pygame.font.SysFont("arial", self._font_size)
            if self._la is not None:
                last_action_row = next((i for i, v in np.ndenumerate(self.board[self._la]) if v == 0), (None, None))[0]
                if last_action_row is not None:
                    last_action = f"{chr(ord('A') + self._la)}{last_action_row}"
                else:
                    raise RuntimeError
            else:
                last_action = 'None'
            turn_count = self.t
            status_text = f"Turn: {turn_count} | Last action: {last_action}"
            text_surface = status_font.render(status_text, True, (0, 0, 0))
            status_rect = pygame.Rect(0, 0, self.window_size[0] + self.index_width, self.status_bar_height)
            pygame.draw.rect(self.window_surface, (255, 255, 255), status_rect)
            text_rect = text_surface.get_rect(center=(status_rect.width // 2, status_rect.height // 2))
            self.window_surface.blit(text_surface, text_rect)
        except Exception:
            pass

        HIGHLIGHT_COLOR = (0, 255, 0)
        BORDER_WIDTH = 4
        PADDING = 4  # pushes rectangle slightly outside discs

        for key, rect_cells in self._rectangles.items():

            # Sort for consistent orientation detection
            rect_cells = sorted(rect_cells)

            # Convert board coordinates to pixel centers
            centers = []
            for col, board_row_idx in rect_cells:
                visual_row = self.column_height - 1 - board_row_idx

                x = (
                        self.index_width
                        + col * self.cell_size[0]
                        + self.cell_size[0] // 2
                )
                y = (
                        self.status_bar_height
                        + visual_row * self.cell_size[1]
                        + self.cell_size[1] // 2
                )

                centers.append((x, y))

            x0, y0 = centers[0]
            x1, y1 = centers[-1]

            dx = x1 - x0
            dy = y1 - y0

            cell_w = self.cell_size[0]
            cell_h = self.cell_size[1]

            # Determine orientation and correct length
            if dy == 0:
                # Horizontal
                angle = 0
                length = 4 * cell_w

            elif dx == 0:
                # Vertical
                angle = 90
                length = 4 * cell_h

            else:
                # Diagonal
                angle = -45 if dx * dy > 0 else 45
                length = 2 * radius + 3 * 2 ** 0.5 * min(self.cell_size) + 8

            thickness = cell_h

            # Add padding so border sits outside discs
            length += PADDING
            thickness += PADDING

            # Create transparent surface
            surf = pygame.Surface((length, thickness), pygame.SRCALPHA)

            pygame.draw.rect(
                surf,
                HIGHLIGHT_COLOR,
                (0, 0, length, thickness),
                BORDER_WIDTH,
                border_radius=thickness // 2,
            )

            # 1. Generate the text surface first to get its dimensions
            text_font = pygame.font.SysFont("arial", self._font_size, bold=True)
            text_surf = text_font.render(str(key), True, (0, 0, 0), HIGHLIGHT_COLOR)
            text_w, text_h = text_surf.get_size()

            # 2. Calculate symmetrical surface dimensions
            # Add text_h to the top AND the bottom to keep the center anchored
            total_thickness = thickness + (2 * text_h)
            total_length = max(length, text_w)  # Ensure surface is wide enough for text

            # 3. Create the transparent surface with the new symmetrical bounds
            surf = pygame.Surface((total_length, total_thickness), pygame.SRCALPHA)

            # 4. Draw the rectangle, offsetting it downward by text_h
            pygame.draw.rect(
                surf,
                HIGHLIGHT_COLOR,
                (0, text_h, length, thickness),
                BORDER_WIDTH,
                border_radius=thickness // 2,
            )

            # 5. Draw the text at the top-left (right above the rectangle)
            surf.blit(text_surf, (thickness // 2, 0))

            # Rotate surface
            rotated = pygame.transform.rotate(surf, angle)

            # Position at midpoint of first and last disc
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2

            # Because we used symmetric padding, the center of 'rotated' is still
            # perfectly aligned with the center of the rectangle itself.
            rect = rotated.get_rect(center=(mid_x, mid_y))

            self.window_surface.blit(rotated, rect)

        # finalize display
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(30)


def cli():
    # Minimal interactive CLI for PyCharm run (no argparse).
    # Play by typing a column index and pressing Enter. Type 'q' to quit.

    env = Connect4Env()

    while True:
        env.reset()
        done = False

        while not done:
            env.render()
            player = env.current_player
            moves = env.legal_actions
            if not moves:
                break

            # show human-readable player number (1 or 2)
            player_num = 1 if player == WHITE else 2
            prompt = f'Player {player_num} turn. Enter column (0-{env.num_columns - 1}) or "q" to quit: '
            choice = input(prompt).strip().lower()
            if choice == 'q':
                print('Quitting.')
                raise SystemExit(0)

            try:
                c = int(choice)
            except ValueError:
                print('Invalid input, please enter a column number.')
                continue

            if c not in moves:
                print(f'Illegal move. Legal moves are: {moves}')
                continue

            _, _, done, _, _ = env.step(c)

        env.render()
        if env.done is None:
            print('Game ended without a winner.')
        elif env.done == 0:
            print('Draw.')
        else:
            # show human-readable winner number
            winner = 'X' if env.done == WHITE else 'O'
            print(f'Player {winner} wins!')

        again = input('Play again? [y/N]: ').strip().lower()
        if again != 'y':
            break


if __name__ == '__main__':
    cli()

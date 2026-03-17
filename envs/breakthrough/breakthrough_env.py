import sys
from copy import deepcopy
from os import path

import numpy as np
import pygame
from gymnasium import Env, spaces

from ..common.constants import *
from ..common import BaseEnv


class BreakthroughException(Exception):
    def __init__(self, message="Generic Breakthrough exception."):
        super().__init__(message)

class InvalidPieceSelectionException(BreakthroughException):
    def __init__(self, message="You can't select this piece."):
        super().__init__(message)

class InvalidActionException(BreakthroughException):
    def __init__(self, message="You can't select this destination square."):
        super().__init__(message)

class IllegalActionException(BreakthroughException):
    def __init__(self, message="This action is illegal."):
        super().__init__(message)

class BreakthroughEnv(BaseEnv, Env):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 4,
    }

    ANSI_PIECES = {
        WHITE: "♙",
        BLACK: "♟",
    }

    DIRECTIONS = ['left', 'straight', 'right']

    LIGHT_COLOR = (245, 245, 220)
    DARK_COLOR = (119, 136, 153)
    HIGHLIGHT_COLOR = (0, 255, 0)

    def __init__(self, max_episode_length=None, **kwargs):
        self.ncol = self.nrow = 8
        self.action_space = spaces.Discrete(self.ncol * self.nrow * 3)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.nrow, self.ncol), dtype=np.uint8)
        self.board = None
        self.done = False
        self.current_player = None
        self.human_color = None
        self.agent_color = None
        self.t = None
        if max_episode_length is None:
            max_episode_length = np.inf
        self._max_episode_length = max_episode_length
        self._la = None
        self._la_lan = None

        self._pieces_positions = None

        self.window_size = (min(64 * self.ncol, 512), min(64 * self.nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_size = self.window_size[0], self.window_size[1]

        self.index_width = 24
        self.status_bar_height = 40

        self.window_surface = None
        self._font_size = 19
        self.clock = None
        self.selected = None
        self.images = {
            WHITE: None,
            BLACK: None,
        }
        self.render_mode = kwargs.pop('render_mode', None)

    def __copy__(self):
        new_env = BreakthroughEnv(max_episode_length=self.max_episode_length, render_mode=self.render_mode)
        new_env.board = deepcopy(self.board)
        new_env.done = self.done
        new_env.current_player = self.current_player
        new_env.human_color = self.human_color
        new_env.agent_color = self.agent_color
        new_env.t = self.t
        new_env._la = self._la
        new_env._pieces_positions = deepcopy(self._pieces_positions)
        return new_env

    @property
    def observation(self):
        return deepcopy(self._pieces_positions), self.current_player

    @property
    def flipped_board(self):
        return np.rot90(self.board, k=2)

    @property
    def other_player(self):
        return self.opponent_color(self.current_player)

    @property
    def white_positions(self):
        return self._pieces_positions[WHITE]

    @property
    def black_positions(self):
        return self._pieces_positions[BLACK]

    @property
    def state_space_cardinality(self):
        return self.nrow * self.ncol

    @property
    def action_space_cardinality(self):
        return 4

    @property
    def max_episode_length(self):
        return self._max_episode_length

    @staticmethod
    def opponent_color(color):
        """
        Returns the opponent color for the given color.
        """
        # p = 1-(p-1)+1 = 3 - p
        assert color is not None
        return 3 - color

    def get_piece_id_from_pos(self, i, j):
        color = self.board[i, j]
        piece_id = next((k for k, v in self._pieces_positions[color].items() if v == (i, j)), None)
        if piece_id is None:
            raise RuntimeError
        return piece_id

    def get_direction(self, d):
        if self.human_color == WHITE:
            return self.DIRECTIONS[d]
        else:
            return list(reversed(self.DIRECTIONS))[d]

    def reward(self):
        if not self.done:
            return 0
        else:
            if self.done == self.human_color:
                return -1
            else:
                return 1

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            super(BreakthroughEnv, self).reset(seed=kwargs['seed'])

        assert 'human_first' in kwargs
        human_first = kwargs['human_first']

        if human_first:
            # Human is white
            self.human_color = WHITE
            self.agent_color = BLACK
        else:
            # Human is black
            self.human_color = BLACK
            self.agent_color = WHITE

        self.board = np.zeros((self.nrow, self.ncol), dtype=np.uint8)
        self.board[0:2, :] = BLACK
        self.board[-2:, :] = WHITE
        self.done = False
        self.current_player = WHITE
        self.t = 0
        self._la = None
        self._la_lan = None

        self._pieces_positions = {WHITE: dict(), BLACK: dict()}

        for i in range(self.nrow):
            for j in range(self.ncol):
                piece = self.board[i, j]
                if piece == EMPTY_CELL:
                    continue
                self._pieces_positions[piece][len(self._pieces_positions[piece])] = (i, j)

        self.window_surface = None
        self.clock = None
        self.selected = None


    def _encode_action(self, i, j, direction):
        """
        Get a single integer from i, j and direction (among 'left', 'straight', 'right').
        """
        return (i * self.ncol + j) * 3 + BreakthroughEnv.DIRECTIONS.index(direction)

    def _decode_action(self, action):
        """
        Get i, j and direction (among 'left', 'straight', 'right') from a single integer.
        """
        return self.decode_action(action, self.ncol)

    @staticmethod
    def decode_action(action, ncol):
        """
        Get i, j and direction (among 'left', 'straight', 'right') from a single integer.
        """
        direction = action % 3
        action -= direction
        action //= 3
        i, j = action // ncol, action % ncol
        return i, j, direction

    def cell_indices_to_name(self, i, j):
        """
        Convert cell indices (i, j) to a human-readable name.
        """
        row_name = self.nrow - i
        col_name = chr(ord('a') + j)

        return f"{col_name}{row_name}"

    def decode_action_human(self, action, lan=True, check_validity=True, player=None):
        """
        Get i, j and direction (among 'left', 'straight', 'right') from a single integer.
        """
        if not check_validity and player is not None:
            current_player = player
            other_player = self.opponent_color(player)
        else:
            current_player = self.current_player
            other_player = self.other_player

        if lan:
            if isinstance(action, int):
                i, j, _ = self._decode_action(action)
                ii, jj = self.compute_dest_cell(self.board, current_player, action, check_validity)
            else:
                i, j, ii, jj = action
            if check_validity and self.board[ii, jj] == other_player:
                capture = 'x'
            else:
                capture = ''
            return self.cell_indices_to_name(i, j) + capture + self.cell_indices_to_name(ii, jj)
        else:
            i, j, direction = self._decode_action(action)
            piece_id = self.get_piece_id_from_pos(i, j)
            piece_direction = self.get_direction(direction)
            piece_color = self.board[i, j]
            return f"{BreakthroughEnv.ANSI_PIECES[piece_color]}{piece_id+1} {piece_direction}"

    def _legal_position(self, i, j):
        return 0 <= i < self.nrow and 0 <= j < self.ncol

    def compute_dest_cell(self, board, player, action, check_validity=True):
        i, j, direction = self._decode_action(action)

        if check_validity and board[i, j] != player:
            raise InvalidPieceSelectionException(f"Player {'WHITE' if player==WHITE else 'BLACK'} cannot select piece at ({i}, {j})")

        d_i = -1 if player == WHITE else 1
        d_j = direction - 1

        ii = i + d_i
        jj = j + d_j

        if check_validity and not self._legal_position(ii, jj):
            raise InvalidActionException

        return ii, jj

    def step(self, action) :
        if not self.action_space.contains(action):
            raise IllegalActionException

        if self.done:
            raise RuntimeError('BreakthroughEnv was terminated')

        i, j, direction = self._decode_action(action)
        (ii, jj) = self.compute_dest_cell(self.board, self.current_player, action)

        dest_cell = self.board[ii, jj]

        self._la_lan = self.decode_action_human((i, j, ii, jj))

        # handle the possibility to eat
        if dest_cell == self.current_player:
            raise InvalidActionException(f"Player {'WHITE' if self.current_player==WHITE else 'BLACK'} cannot move piece from ({i}, {j}) to ({ii}, {jj}) because the destination cell is occupied by another piece of the same color.")

        info = {}
        if dest_cell != EMPTY_CELL:
            if self.DIRECTIONS[direction] == 'straight':
                raise InvalidActionException
            else:
                info['captured'] = True
                captured_id = self.get_piece_id_from_pos(ii, jj)
                assert captured_id is not None
                self._pieces_positions[self.other_player][captured_id] = None

        # update the dictionaries
        moved_piece_id = self.get_piece_id_from_pos(i, j)
        self._pieces_positions[self.current_player][moved_piece_id] = (ii, jj)

        # update the board
        self.board[i, j] = EMPTY_CELL
        self.board[ii, jj] = self.current_player

        # check for the game to be over
        if self.current_player == WHITE and ii == 0:
            self.done = WHITE
        elif self.current_player == BLACK and ii == self.nrow-1:
            self.done = BLACK
        elif len(list(filter(lambda p: p is not None, self._pieces_positions[WHITE].values()))) == 0:
            self.done = BLACK
        elif len(list(filter(lambda p: p is not None, self._pieces_positions[BLACK].values()))) == 0:
            self.done = WHITE

        # change turn
        self.current_player = self.other_player

        self.t += 1
        truncated = True if self.t >= self.max_episode_length else False
        self._la = action

        return self.observation, self.reward(), self.done, truncated, info

    def render(self):
        if self.render_mode == 'ansi':
            print(f"Current board ({self.ANSI_PIECES[self.other_player]} turn):")
            if self.human_color == WHITE:
                board = self.board
            else:
                board = self.board[::-1, :]
            for row in board:
                print(" ".join(self.ANSI_PIECES[x] if x != 0 else "." for x in row))
            print()
        elif self.render_mode == 'human':
            return self._render_gui(self.render_mode)
        else:
            raise NotImplementedError

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise ImportError("pygame is not installed. Run `pip install pygame`.") from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Breakthrough")
                # Add space for index column, row, and top status bar
                self.window_surface = pygame.display.set_mode(
                    (self.window_size[0] + self.index_width, self.window_size[1] + self.index_width + self.status_bar_height)
                )

        assert self.window_surface is not None, "Pygame window surface creation failed."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window_surface.fill((50, 50, 50))

        if self.human_color == WHITE:
            board = self.board
        else:
            board = self.flipped_board

        # Draw board with offset for index column/row and status bar
        for row in range(self.nrow):
            for col in range(self.ncol):
                pos = (
                    self.index_width + col * self.cell_size[0],
                    self.status_bar_height + row * self.cell_size[1]
                )
                rect = (*pos, *self.cell_size)
                color = self.LIGHT_COLOR if (row + col) % 2 == 0 else self.DARK_COLOR
                pygame.draw.rect(self.window_surface, color, rect)

                if self.selected and self.selected == (row, col):
                    pygame.draw.rect(self.window_surface, self.HIGHLIGHT_COLOR, rect, 5)

                piece = board[row][col]
                if piece:
                    if self.images[WHITE] is None:
                        file_name = path.join(path.dirname(path.dirname(__file__)), "img/pawn_white.png")
                        self.images[WHITE] = pygame.transform.scale(
                            pygame.image.load(file_name), self.cell_size
                        )
                    if self.images[BLACK] is None:
                        file_name = path.join(path.dirname(path.dirname(__file__)), "img/pawn_black.png")
                        self.images[BLACK] = pygame.transform.scale(
                            pygame.image.load(file_name), self.cell_size
                        )

                    self.window_surface.blit(self.images[piece], pos)

                    if self.human_color == WHITE:
                        r, c = row, col
                    else:
                        r = self.nrow - row - 1
                        c = self.ncol - col - 1

                    piece_id = self.get_piece_id_from_pos(r, c)
                    text_color = (0, 0, 0) if piece == WHITE else (255, 255, 255)
                    id_font = pygame.font.SysFont("arial", self._font_size, bold=True)
                    text_surface = id_font.render(str(piece_id + 1), True, text_color)
                    pos_arr = np.array(pos)
                    text_size = np.array(text_surface.get_size())
                    cell_size = np.array(self.cell_size)
                    text_pos = pos_arr + (cell_size - text_size) / 2
                    text_pos[1] -= 1
                    self.window_surface.blit(text_surface, text_pos)

        col_font = pygame.font.SysFont("arial", self._font_size, bold=True)

        if self.human_color == WHITE:
            col_labels = [chr(ord('A') + i) for i in range(self.ncol)]
            row_labels = list(reversed(range(1, self.nrow + 1)))
        else:
            col_labels = [chr(ord('A') + i) for i in range(self.ncol)][::-1]
            row_labels = list(range(1, self.nrow + 1))

        # Column indices (letters)
        for col, label in zip(range(self.ncol), col_labels):
            text_surface = col_font.render(label, True, (255, 255, 255))
            x = self.index_width + col * self.cell_size[0] + self.cell_size[0] // 2 - text_surface.get_width() // 2
            y = self.status_bar_height + self.nrow * self.cell_size[1] + (self.index_width // 2 - text_surface.get_height() // 2)
            self.window_surface.blit(text_surface, (x, y))

        # Row indices (numbers)
        row_font = pygame.font.SysFont("arial", self._font_size, bold=True)
        for row, label in zip(range(self.nrow), row_labels):
            text_surface = row_font.render(str(label), True, (255, 255, 255))
            x = self.index_width // 2 - text_surface.get_width() // 2
            y = self.status_bar_height + row * self.cell_size[1] + self.cell_size[1] // 2 - text_surface.get_height() // 2
            self.window_surface.blit(text_surface, (x, y))

        # Status bar at the top
        status_font = pygame.font.SysFont("Apple Symbols", 25)
        status_bar_rect = pygame.Rect(
            0,
            0,
            self.window_size[0] + self.index_width,
            self.status_bar_height
        )
        pygame.draw.rect(self.window_surface, (240, 240, 240), status_bar_rect)

        # Count pieces
        white_count = sum(cell == WHITE for row in board for cell in row)
        black_count = sum(cell == BLACK for row in board for cell in row)

        # Get turn count (assuming you track it)
        turn_count = self.t

        # Prepare status text
        if self.selected:
            r, c = self.selected
            if self.human_color == WHITE:
                pass
            elif self.human_color == BLACK:
                r = self.nrow - r - 1
                c = self.ncol - c - 1
            selected_piece = self.get_piece_id_from_pos(r, c)
        else:
            selected_piece = ""

        last_action = self._la_lan if self._la_lan is not None else 'None'
        status_text = f"{self.ANSI_PIECES[WHITE]}: {white_count} | {self.ANSI_PIECES[BLACK]}: {black_count} | Turn: {turn_count} | Sel.: {selected_piece} | Last action: {last_action}"
        text_surface = status_font.render(status_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=((self.window_size[0] + self.index_width) // 2, self.status_bar_height // 2))
        self.window_surface.blit(text_surface, text_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    @property
    def legal_actions(self):
        """
        This property calls the utility method legal_actions_board with the current board and player.
        """
        return self.legal_actions_board(self.board, self.current_player)


    def legal_actions_board(self, board, player):
        """
        This is a utility method that computes the legal actions for a given board and player.
        """
        la = []
        for i in range(self.nrow):
            for j in range(self.ncol):
                if board[i, j] != player:
                    continue
                for d_idx, direction in enumerate(self.DIRECTIONS):
                    d_j = d_idx - 1
                    d_i = -1 if player == WHITE else 1

                    ii = i + d_i
                    jj = j + d_j

                    if self._legal_position(ii, jj):
                        dest_cell = board[ii, jj]
                        if d_j == 0 and dest_cell == 0:
                            # can go straight only if the cell is empty
                            la.append(self._encode_action(i, j, direction))
                        elif d_j != 0 and dest_cell != self.current_player:
                            # can go diagonally only if empty cell or opponent piece
                            la.append(self._encode_action(i, j, direction))

        return la

    @property
    def _last_action(self):
        return self._la

    @property
    def adversarial(self):
        return True

    @staticmethod
    def build_board_from_obs(obs, nrow, ncol):
        board = np.zeros((nrow, ncol), dtype=np.uint8)
        for color, positions in obs.items():
            for piece_id, pos in positions.items():
                if pos is None:
                    continue
                board[pos[0], pos[1]] = color

        return board

    def backup(self):
        state = {
            'state': self.observation,
            'board': deepcopy(self.board),
            'done': self.done,
            'last_action': self._last_action,
            'last_action_lan': self._la_lan,
            't': self.t,
            'reward': self.reward(),
            'pieces_positions': deepcopy(self._pieces_positions),
            'current_player': self.current_player,
            'player': 'Human' if self.current_player == self.human_color else 'Agent',
        }
        return state

    def load(self, checkpoint):
        try:
            pieces, player = checkpoint['state']
            self.board = self.build_board_from_obs(pieces, self.nrow, self.ncol)
            self.done = checkpoint['done']
            self._la = checkpoint['last_action']
            self._la_lan = checkpoint['last_action_lan']
            self.current_player = player
            self.t = checkpoint['t']
            self._pieces_positions = pieces
        except KeyError as e:
            print(e, file=sys.stderr)
            return False
        return True

    def game_result(self, human_readable=False):
        if human_readable:
            if not self.done:
                s = "Game still running..."
            else:
                if self.done == self.human_color:
                    s = "You won!"
                else:
                    s = "You lost!"

        else:
            if not self.done:
                s = 0
            else:
                if self.done == self.human_color:
                    s = 1
                else:
                    s = 0

        return s

    def get_square_under_mouse(self, pos):
        j, i = pos
        return (i-self.status_bar_height) // self.cell_size[1], (j-self.index_width) // self.cell_size[0]

    def get_mouse_action(self):
        if self.human_color == WHITE:
            board = self.board
        else:
            board = self.flipped_board
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                row, col = self.get_square_under_mouse(pygame.mouse.get_pos())
                if self.selected:
                    selected_row, selected_col = self.selected
                    piece = board[selected_row][selected_col]
                    if (
                            selected_row == row  # impossible to move on the same row
                            or (piece == self.human_color and row != selected_row - 1)  # only possible to move one cell vertically
                            or (piece == self.agent_color and row != selected_row + 1)  # only possible to move one cell vertically
                            or col not in [selected_col - 1, selected_col, selected_col + 1]  # illegal horizontal destination square
                            or not (0 <= col <= self.ncol - 1)  # illegal horizontal destination square
                            or piece == board[row][col]  # impossible to move if there's another piece of the same color
                    ):
                        self.selected = None
                        self.render()
                        return None

                    # if we make it here, the move is legal
                    jj = col - selected_col + 1
                    direction = self.DIRECTIONS[jj]
                    self.selected = None
                    if self.human_color == BLACK:
                        # i.e. if the board is flipped
                        selected_row = self.nrow - 1 - selected_row
                        selected_col = self.ncol - 1 - selected_col
                        direction = self.DIRECTIONS[-(1+jj)]

                    return self._encode_action(selected_row, selected_col, direction)

                elif board[row][col]:
                    self.selected = (row, col)
                    if self.render_mode == 'human':
                        self.render()

    @staticmethod
    def board_is_terminal(board):
        """
        The board is terminal either if one player has reached the opponent's baseline
        or if one player has no pieces left.
        """
        # Check if a player has reached opponent's baseline
        if np.any(board[0, :] == WHITE) or np.any(board[-1, :] == BLACK):
            return True

        # Check if a player has no pieces left
        if not np.any(board == WHITE):
            return True

        if not np.any(board == BLACK):
            return True


def human_main():
    env = BreakthroughEnv(render_mode="human")
    env.reset(human_first=False)
    env.render()

    done = False
    while not done:
        action = None
        while action is None:
            action = env.get_mouse_action()

        try:
            obs, rew, done, _, _ = env.step(action)
        except BreakthroughException as e:
            print(e, file=sys.stderr)
        env.render()

    print(env.game_result())


def console_main():
    env = BreakthroughEnv(render_mode="human")
    env.reset(human_first=True)
    env.render()

    done = False
    while not done:
        action = input('Action: ')
        # ps = action.split(' ')
        # i = int(ps[0])
        # j = int(ps[1])
        # direction = ps[2]
        # action = env._encode_action(i, j, direction)
        action = int(action)
        obs, rew, done, _, _ = env.step(action)
        env.render()

    print(env.game_result())


if __name__ == '__main__':
    # human_main()
    console_main()
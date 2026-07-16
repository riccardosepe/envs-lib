import sys

import numpy as np

from ..breakthrough.breakthrough_env import BreakthroughEnv, BreakthroughException
from ..common.constants import *


class VectorBreakthroughEnv(BreakthroughEnv):
    """
    A Breakthrough variant that differs from the standard game only in its
    terminal condition and in *who* wins when the game ends.

    Everything else — movement/capture rules, board setup, action encoding,
    rendering, checkpointing and MCTS backup/load — is inherited unchanged
    from :class:`BreakthroughEnv`.

    The single override point is :meth:`_winner`: it returns the *winning
    color* (``WHITE`` or ``BLACK``), which may differ from ``mover`` (the
    player who just moved). Returning a color ends the game and, via
    ``self._done``, drives ``reward()`` / ``game_result()`` automatically —
    so the "player who ends the game" and the "player who wins" can differ.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scores = np.array([2, 1, 0, 0, 0, 0, 0, 0])

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.board = np.zeros((self.nrow, self.ncol), dtype=np.uint8)
        self.board[1:2, :] = BLACK
        self.board[-5:-4, :] = WHITE
        self._pieces_positions = {WHITE: dict(), BLACK: dict()}

        for i in range(self.nrow):
            for j in range(self.ncol):
                piece = self.board[i, j]
                if piece == EMPTY_CELL:
                    continue
                self._pieces_positions[piece][len(self._pieces_positions[piece])] = (i, j)

    def _winner(self, mover, dest_row):
        """
        Determine the winner after ``mover`` moved a piece to row ``dest_row``.

        Return ``WHITE`` or ``BLACK`` if the game is over, else ``None``.
        The full env state is available on ``self`` (e.g. ``self.board``,
        ``self._pieces_positions``, ``self.nrow`` / ``self.ncol``) if the
        condition needs more than ``mover`` and ``dest_row``.
        """
        if not self.legal_actions_board(self.board, self.other_player):
            # NB: the default representation of a board is white below, black above
            white_score = (np.where(self.board == WHITE, 1, 0) * self._scores).sum()
            black_score = (np.where(self.board == BLACK, 1, 0) * self._scores[::-1]).sum()
            if white_score > black_score:
                return WHITE
            elif white_score == black_score:
                return DRAW
            else:
                return BLACK
        else:
            return None

    @property
    def reward_space_cardinality(self):
        """Number of components in the reward vector.

        Required by :class:`VMCTS` (it seeds the root reward with a zero vector
        of this length). Set it to however many objectives your reward will
        decompose into.
        """
        return 2

    def reward(self):
        """Vector reward for the last transition.

        Must return an ``np.ndarray`` of shape ``(reward_space_cardinality,)``.
        See :meth:`SailingDomainEnv._compute_reward` for a worked multi-objective
        example. The terminal outcome is available via ``self.done`` (``WHITE`` /
        ``BLACK`` / ``DRAW``) and the positional weights via ``self._scores``.

        For adversarial search with :class:`VMCTS`, express this as a *zero-sum*
        vector from the perspective of the player who just moved, so that the
        negamax sign-flip in ``VMCTS._backpropagate`` yields correct minimax.
        """
        if not self.done:
            return np.zeros(self.reward_space_cardinality)

        white_score_1 = np.count_nonzero(self.board[0] == WHITE)
        white_score_2 = np.count_nonzero(self.board[1] == WHITE)
        black_score_1 = np.count_nonzero(self.board[-1] == BLACK)
        black_score_2 = np.count_nonzero(self.board[-2] == BLACK)
        reward_vector = np.array([white_score_1 - black_score_1, white_score_2 - black_score_2])
        if self.current_player == BLACK:
            reward_vector = -reward_vector
        return reward_vector

    def game_result(self, human_readable=False):
        if self.done == DRAW:
            return "It's a draw!" if human_readable else 0
        return super().game_result(human_readable=human_readable)

    @staticmethod
    def board_is_terminal(board):
        # NB: BreakthroughEnv.board_is_terminal (a @staticmethod returning just a
        #  bool) has no callers in this codebase, so it is intentionally not
        #  overridden. If you later wire it into MCTS, override it here too and keep
        #  it consistent with _winner.
        raise NotImplementedError


def human_main():
    env = VectorBreakthroughEnv(render_mode="human")
    env.reset(agent_color=BLACK)
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
    env = VectorBreakthroughEnv(render_mode="human")
    env.reset(agent_color=WHITE)
    env.render()

    done = False
    while not done:
        action = input('Action: ')
        action = int(action)
        obs, rew, done, _, _ = env.step(action)
        env.render()

    print(env.game_result())


if __name__ == '__main__':
    # console_main()
    human_main()

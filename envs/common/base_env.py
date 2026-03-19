from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    This class is to be intended in the sense of a Java interface. All the custom environments must both inherit
    gym(nasium).Env and implement this interface. In practice, the interface standardizes all the methods that are
    needed in the current framework.

    Backup / load contract
    ----------------------
    The checkpoint dict is split into two layers:
      1. Common keys — written and read by BaseEnv.backup() / BaseEnv.load():
             'done', 't', 'last_action', 'reward', 'player'
      2. Subclass-specific keys — written and read by the overriding backup() / load()
         after calling super().

    Subclasses MUST call super().backup() / super().load() first, then extend the
    returned dict / restore additional attributes.
    """

    WON = 0
    LOST = 1
    STILL_RUNNING = 2

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def max_episode_length(self):
        pass

    @property
    @abstractmethod
    def legal_actions(self):
        pass

    @property
    @abstractmethod
    def _last_action(self):
        pass

    @property
    @abstractmethod
    def adversarial(self):
        pass

    @property
    @abstractmethod
    def state_space_cardinality(self):
        """Number of states in the environment."""
        pass

    @property
    @abstractmethod
    def action_space_cardinality(self):
        """Number of actions in the environment."""
        pass

    # ------------------------------------------------------------------
    # Concrete properties
    # ------------------------------------------------------------------

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
        self._done = value

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def game_result(self, **kwargs):
        pass

    @abstractmethod
    def reward(self):
        """
        Return the reward associated with the *current* state.

        For environments where the reward depends on the transition
        (s, a, s'), the step() method is responsible for computing and
        storing the reward internally so that this method can return it
        without requiring any arguments.

        Return type is intentionally flexible:
          - adversarial envs return a scalar (float / int)
          - multi-objective envs (e.g. SailingDomain) return a vector (np.ndarray)
        """
        pass

    # ------------------------------------------------------------------
    # Shared player utility (adversarial envs)
    # ------------------------------------------------------------------

    @staticmethod
    def opponent_color(color):
        """
        Returns the opponent color for the given color.
        Works for any two-player game that encodes players as 1 and 2
        (WHITE = 1, BLACK = 2 by the common constants convention).
        """
        assert color is not None
        return 3 - color

    # ------------------------------------------------------------------
    # Shared backup / load
    # ------------------------------------------------------------------

    def backup(self):
        """
        Snapshot the common MCTS metadata fields.

        Subclasses extend this by calling super().backup() and then
        adding their own environment-specific fields:

            def backup(self):
                state = super().backup()
                state['board'] = deepcopy(self.board)
                state['my_extra_field'] = self.my_extra_field
                return state
        """
        return {
            'done': self.done,
            't': self.t,
            'last_action': self._last_action,
            'reward': self.reward(),
            'player': self._player_label(),
        }

    def load(self, checkpoint):
        """
        Restore the common MCTS metadata fields.

        Subclasses extend this by calling super().load() first and then
        restoring their own environment-specific fields:

            def load(self, checkpoint):
                super().load(checkpoint)
                self.board = deepcopy(checkpoint['board'])
                self.my_extra_field = checkpoint['my_extra_field']

        Raises KeyError if a required common key is missing (callers
        should wrap this in try/except as appropriate for their context).
        """
        self.done = checkpoint['done']
        self.t = checkpoint['t']
        self._la = checkpoint['last_action']
        # 'reward' is intentionally not restored as live state — it is
        # only stored in the checkpoint for MCTS bookkeeping purposes.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _player_label(self):
        """
        Returns a human-readable string indicating who is to move.
        Adversarial envs override this if they track human_color;
        single-agent envs always return 'Agent'.
        """
        return 'Agent'

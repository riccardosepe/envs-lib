from abc import ABC, abstractmethod


# ==============================================================================
# Framework-level exception
# ==============================================================================

class EnvStepException(Exception):
    """
    Base exception for illegal or invalid actions passed to env.step().

    Raising this from within step() signals to the game loop that the
    move was rejected and the turn should NOT advance.  Domain-specific
    exceptions (e.g. BreakthroughException) must inherit from this class
    so that the loop can catch them with a single except clause.
    """
    def __init__(self, message="Illegal or invalid action."):
        super().__init__(message)


# ==============================================================================
# Base environment interface
# ==============================================================================

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

    Checkpoint contract
    -------------------
    Subclasses that support resuming from a saved game state MUST implement
    load_checkpoint(checkpoint_id).  Subclasses that do NOT support checkpoints
    (e.g. single-agent envs with no save format) should leave the default
    implementation in place, which raises NotImplementedError.

    The classmethod signature is:

        @classmethod
        def load_checkpoint(cls, checkpoint_id: int) -> tuple[dict, int | None]:
            ...

    Return value: (ckpt_dict, agent_color)
      - ckpt_dict    : the state dict to be passed to env.load()
      - agent_color  : WHITE or BLACK for adversarial envs, None for single-agent

    The caller (main._build_env) is responsible for calling env.reset() and
    env.load(ckpt_dict) after receiving these values.
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

    # ------------------------------------------------------------------
    # Input label hook
    # ------------------------------------------------------------------

    def decode_action_input(self, action) -> str:
        """
        Return the string a human should type to select this action.

        Defaults to decode_action_human(), which is correct for most envs.
        Override when the display label carries extra information that
        would be tedious to retype (e.g. Connect4 shows "A3" for display
        but only needs "A" for input, since the landing row is determined
        by the current board state and is not part of the move itself).

        Must be consistent with get_user_action()'s matching logic: the
        returned string is compared case-insensitively against raw input.
        """
        return self.decode_action_human(action)

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
    # Checkpoint classmethod
    # ------------------------------------------------------------------

    @classmethod
    def load_checkpoint(cls, checkpoint_id: int):
        """
        Load a previously saved game state by its integer ID.

        Subclasses that support checkpoints MUST override this method.
        The default implementation raises NotImplementedError so that
        outdated or checkpoint-unaware envs fail loudly rather than
        silently passing through an ID that is ignored.

        Parameters
        ----------
        checkpoint_id : int
            An integer key identifying the saved state (e.g. a slot
            number, an episode index, or a database row id).

        Returns
        -------
        ckpt_dict : dict
            State snapshot suitable for passing to env.load().
        agent_color : int or None
            WHITE or BLACK for adversarial envs; None for single-agent.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented checkpoint support.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement load_checkpoint(). "
            "Either add checkpoint support or do not pass a checkpoint_id."
        )

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

    def decode_action_human(self, action):
        return NotImplementedError
